"""Extract shared frozen features and masked/clean features for the day-held-out experiments."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from audit_gpu_yolo import targets
from features import BOX, CLASSES, D, load
from audit_adapt_dino import WORK, SIZE

CACHE = WORK / "hypotheses_v1"
CACHE.mkdir(exist_ok=True)
SCALE = np.array([1280, 720, 1280, 720])
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def metadata():
    data, excluded = [], set()
    for part in ("train", "test"):
        det, _, ctx, _ = load(part)
        excluded.update(ctx[["frame_id", "previous", "next"]].stack())
        data.append(det[det.class_name.isin(CLASSES)].assign(part=part))
    raw = pd.read_csv(D / "human_boxes_train.csv").dropna(subset=["class_name", "w", "h"])
    raw["class_name"] = raw.class_name.replace({"nude_hand": "nude_hands"})
    raw[BOX] = np.column_stack([
        raw.cx - raw.w / 2, raw.cy - raw.h / 2, raw.cx + raw.w / 2, raw.cy + raw.h / 2
    ]) * SCALE
    raw = raw[(raw.w > 0) & (raw.h > 0)].copy()
    return pd.concat(data, ignore_index=True), raw.reset_index(drop=True), excluded


def context(box, rng):
    wh = np.maximum(box[2:] - box[:2], 8)
    center = (box[:2] + box[2:]) / 2 + rng.uniform(-.23, .23, 2) * wh
    half = np.maximum(wh * rng.uniform(.84, 1.16, 2), 19)
    return np.r_[center - half, center + half]


def crop(image, bounds):
    bounds = np.round(bounds * np.tile(np.array(image.shape[1::-1]) / [1280, 720], 2)).astype(int)
    x1, y1, x2, y2 = bounds
    h, w = image.shape[:2]
    tile = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if not tile.size:
        raise ValueError(f"Empty crop: {bounds}, image {image.shape}")
    tile = cv2.copyMakeBorder(tile, max(-y1, 0), max(y2-h, 0), max(-x1, 0), max(x2-w, 0),
                             cv2.BORDER_CONSTANT, value=(108, 116, 124))
    return cv2.resize(tile, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()


def read_frame(frame):
    image = cv2.imread(str(D / f"{frame}.jpg"))
    if image is None:
        raise FileNotFoundError(frame)
    return image


def examples():
    det, gt, _ = metadata()
    tables = {f: q[BOX].to_numpy() for f, q in gt[gt.class_name == "table"].groupby("frame_id")}
    rows = pd.concat([det, gt[gt.class_name != "table"].assign(part="gt")], ignore_index=True)
    rng = np.random.default_rng(592)
    entries, tiles = [], []
    for frame, group in rows.groupby("frame_id", sort=False):
        image = read_frame(frame)
        for _, row in group.iterrows():
            box = row[BOX].to_numpy(dtype=float)
            # Train geometry on multiple coordinate systems to reduce shortcuts from centered crops.
            copies = 3 if row.part == "gt" and row.class_name == "person" else 1
            for view in range(copies):
                ctx = context(box, rng)
                if copies > 1:
                    wh = box[2:] - box[:2]
                    shift = np.tile(rng.uniform(-.22, .22, 2) * wh, 2)
                    ctx += shift
                tile = crop(image, ctx)
                margin = (box[2:] - box[:2]) * .12
                tight = crop(image, box + np.r_[-margin, margin])
                entries.append(dict(row, view=view, occluded=False, **dict(zip("abcd", ctx))))
                tiles.append((tile, tight))
                if copies > 1:
                    # Cover the lower body with a crop from an annotated table in the same frame.
                    region = tables.get(frame)
                    if region is None or not len(region):
                        continue
                    overlaps = np.maximum(0, np.minimum(region[:, 2:], box[2:]) -
                                          np.maximum(region[:, :2], box[:2])).prod(1)
                    texture_box = region[np.argmin(overlaps)]
                    trim = (texture_box[2:] - texture_box[:2]) * .28
                    texture = crop(image, texture_box + np.r_[trim, -trim])
                    loc = (box - np.tile(ctx[:2], 2)) / np.tile(ctx[2:] - ctx[:2], 2) * SIZE
                    x0 = max(0, int(loc[0] - rng.uniform(0, 18)))
                    x1 = min(SIZE, int(loc[2] + rng.uniform(0, 18)))
                    y0 = max(0, int(loc[1] + rng.uniform(.36, .70) * (loc[3] - loc[1])))
                    y1 = min(SIZE, int(loc[3] + rng.uniform(3, 22)))
                    if x1 > x0 and y1 > y0:
                        covered = tile.copy()
                        covered[y0:y1, x0:x1] = cv2.resize(texture, (x1-x0, y1-y0))
                        entries.append(dict(row, view=view, occluded=True, **dict(zip("abcd", ctx))))
                        tiles.append((covered, tight))
    return pd.DataFrame(entries), np.asarray(tiles)


def pool_examples():
    det, _, excluded = metadata()
    machine = pd.read_csv(D / "machine_predictions.csv")
    machine["class_name"] = machine.class_name.replace({"nude_hand": "nude_hands"})
    machine["day"] = machine.frame_id.str[-15:-7]
    frames = pd.read_csv(D / "machine_frames.csv")
    frames["day"] = frames.frame_id.str[-15:-7]
    eligible = frames[frames.day.isin(["20261025", "20261026"]) & ~frames.frame_id.isin(excluded) &
                      frames.image_exists]
    machine = machine[machine.frame_id.isin(eligible.frame_id) & machine.class_name.isin(CLASSES)]
    crops = []
    for (day, cls), group in machine.groupby(["day", "class_name"]):
        crops.append(group.sample(min(200, len(group)), random_state=146))
    rng = np.random.default_rng(53)
    backgrounds = []
    for day, group in eligible.groupby("day"):
        for frame in group.sample(min(180, len(group)), random_state=74).frame_id:
            wh = rng.uniform([65, 55], [320, 235])
            center = rng.uniform(wh / 2, [1280, 720] - wh / 2)
            backgrounds.append(dict(frame_id=frame, day=day, class_name="background",
                                    **dict(zip(BOX, np.r_[center-wh/2, center+wh/2]))))
    pool = pd.concat([*crops, pd.DataFrame(backgrounds)], ignore_index=True)
    tiles = np.empty((len(pool), SIZE, SIZE, 3), dtype=np.uint8)
    for frame, group in pool.groupby("frame_id", sort=False):
        image = read_frame(frame)
        for idx, row in group.iterrows():
            tiles[idx] = crop(image, context(row[BOX].to_numpy(dtype=float), rng))
    return pool, tiles


def backbone():
    return timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True,
                             num_classes=0, img_size=SIZE).eval().cuda()


@torch.inference_mode()
def extract(model, images, kind):
    mean = torch.tensor(MEAN, device="cuda")[None, :, None, None]
    std = torch.tensor(STD, device="cuda")[None, :, None, None]
    features, reference, masks = [], [], []
    rng = torch.Generator(device="cuda").manual_seed(517)
    for start in range(0, len(images), 32):
        batch = torch.from_numpy(images[start:start+32]).cuda().permute(0, 3, 1, 2) / 255
        batch = (batch - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "masked":
                # Mask pixels before the frozen encoder so the student cannot see the target patches.
                mask = torch.rand(len(batch), 10, 10, device="cuda", generator=rng) < .40
                patch_mask = mask.repeat_interleave(14, 1).repeat_interleave(14, 2)[:, None]
                inputs = torch.cat([batch, batch.masked_fill(patch_mask, 0)])
            else:
                inputs = batch
            x = model.norm_pre(model._pos_embed(model.patch_embed(inputs)))
            early = model.blocks[:-2](x)
            if kind == "early":
                features.append(early.half().cpu())
            elif kind == "masked":
                final = model.norm(model.blocks[-2:](early[:len(batch)]))
                reference.append(final[:, -100:].half().cpu())
                features.append(early[len(batch):].half().cpu())
                masks.append(mask.flatten(1).cpu())
            else:
                final = model.norm(model.blocks[-2:](early))
                features.append(torch.cat([final[:, 0], final[:, -100:].mean(1)], dim=1).half().cpu())
        if start % 256 == 0:
            print(kind, start + len(batch), "/", len(images), flush=True)
    result = torch.cat(features)
    if kind == "masked":
        return result, torch.cat(reference), torch.cat(masks)
    return result


if __name__ == "__main__":
    import sys
    torch.set_num_threads(3)
    cv2.setNumThreads(0)
    net = backbone()
    if sys.argv[-1] != "pool":
        rows, tiles = examples()
        rows.to_csv(CACHE / "rows.csv", index=False)
        torch.save(extract(net, np.ascontiguousarray(tiles[:, 0]), "early"), CACHE / "context.pt")
        torch.save(extract(net, np.ascontiguousarray(tiles[:, 1]), "semantic"), CACHE / "tight.pt")
    if sys.argv[-1] != "labeled":
        rows, tiles = pool_examples()
        rows.to_csv(CACHE / "pool.csv", index=False)
        torch.save(extract(net, tiles, "masked"), CACHE / "pool_masked.pt")
