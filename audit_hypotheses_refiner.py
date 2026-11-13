"""Add synthetic tabletop occlusion to the same refiner used in the 0.90734 baseline."""
import numpy as np
import pandas as pd
import torch
import cv2
from torch.nn import functional as F
from audit_adapt_dino import candidates
from audit_box_refinement import Refiner, supervision
from audit_hypotheses_features import CACHE, WORK, SIZE, metadata, read_frame, backbone, extract
from audit_hypotheses_visual import train_masks
from audit_hypotheses_scene import DAYS
from features import BOX, CLASSES


def paired_tokens(rows, gt, overlap, target):
    rng = np.random.default_rng(92)
    eligible = (rows.part != "test") & (rows.class_name == "person") & (overlap >= .25)
    tables = {f: q[BOX].to_numpy() for f, q in gt[gt.class_name == "table"].groupby("frame_id")}
    patches = []
    source = []
    for frame, group in rows[eligible].groupby("frame_id", sort=False):
        image = read_frame(frame)
        for idx, row in group.iterrows():
            box = row[BOX].to_numpy(dtype=float)
            wh = box[2:] - box[:2]
            lo = np.maximum(0, (box[:2] - .25 * wh).astype(int))
            hi = np.minimum([1280, 720], np.ceil(box[2:] + .25 * wh).astype(int))
            tile = cv2.resize(image[lo[1]:hi[1], lo[0]:hi[0]], (SIZE, SIZE))
            human = box + target[idx] * np.tile(wh, 2)
            box_on_tile = (human - np.tile(lo, 2)) / np.tile(hi-lo, 2) * SIZE
            x1 = max(0, int(box_on_tile[0] - rng.uniform(0, 24)))
            x2 = min(SIZE, int(box_on_tile[2] + rng.uniform(0, 24)))
            y1 = max(0, int(box_on_tile[1] + rng.uniform(.4, .75) * (box_on_tile[3] - box_on_tile[1])))
            y2 = min(SIZE, int(box_on_tile[3] + rng.uniform(3, 24)))
            if x2 <= x1 + 4 or y2 <= y1 + 4 or frame not in tables:
                continue
            region = tables[frame]
            intersects = np.maximum(0, np.minimum(region[:, 2:], human[2:]) -
                                     np.maximum(region[:, :2], human[:2])).prod(1)
            candidates = region[intersects == 0]
            if not len(candidates):
                continue
            table = candidates[rng.integers(len(candidates))]
            wh_t = table[2:] - table[:2]
            u, v, r, b = (table + np.r_[.22*wh_t, -.22*wh_t]).astype(int)
            texture = image[max(0,v):min(b,720), max(0,u):min(r,1280)]
            if texture.size == 0:
                continue
            tile[y1:y2, x1:x2] = cv2.resize(texture, (x2-x1, y2-y1))
            source.append(idx)
            patches.append(tile[:, :, ::-1].copy())
    return np.asarray(source), np.stack(patches)


def fit_predict(tokens, augment, lookup, cls, target, overlap, errors, weight, fit, pred, seed, fraction):
    torch.manual_seed(seed)
    rng = torch.Generator(device="cuda").manual_seed(seed+552)
    model = Refiner(tokens.shape[-1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.03)
    for _ in range(15):
        for idx in fit[torch.randperm(len(fit), device="cuda")].split(64):
            features = tokens[idx]
            replacement = lookup[idx]
            covered = (replacement >= 0) & (torch.rand(len(idx), generator=rng, device="cuda") < fraction)
            features[covered] = augment[replacement[covered]]
            logits = model(features, cls[idx])
            valid = overlap[idx] >= .20
            geom = F.smooth_l1_loss(logits[:, :4], target[idx], reduction="none").mean(1)
            obj = F.binary_cross_entropy_with_logits(logits[:,4], (overlap[idx]>.2).float(), reduction="none")
            err = F.binary_cross_entropy_with_logits(logits[:,5], errors[idx], reduction="none")
            loss = (weight[idx]*(geom*valid*4+obj*.4+err)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.inference_mode():
        return torch.cat([model.eval()(tokens[b], cls[b]) for b in pred.split(64)]).cpu().numpy()


def score(rows, predictions):
    from features import iou_matrix
    bb = rows[BOX].to_numpy()
    revised = (bb + predictions[:,:4] * np.tile(bb[:,2:]-bb[:,:2], 2)).clip([0,0,0,0],[1280,720,1280,720])
    revised[:,2:] = np.maximum(revised[:,2:], revised[:,:2]+1e-6)
    return 1 - np.diag(iou_matrix(bb,revised))/(1 + np.exp(-predictions[:,4]))


def main():
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("high")
    det, gt, _ = metadata()
    truth = gt[gt.class_name.isin(CLASSES)]
    rows = pd.concat([det, candidates(truth).assign(part="synth")], ignore_index=True)
    xy, ov = supervision(rows, truth)
    path = CACHE / "occluded_refiner.pt"
    if not path.exists():
        indices, images = paired_tokens(rows, gt, ov, xy)
        print("Occluded training variants", len(indices), flush=True)
        net = backbone()
        patch = extract(net, images, "early")
        torch.save((torch.from_numpy(indices), patch), path)
        del net
    indices, patch = torch.load(path, weights_only=True)
    tokens = torch.load(WORK / "dino_tokens.pt", weights_only=True).cuda()
    assert len(tokens) == len(rows)
    augment = patch.cuda()
    lookup = torch.full((len(rows),), -1, device="cuda", dtype=torch.long)
    lookup[indices] = torch.arange(len(indices), device="cuda")
    cls = F.one_hot(torch.tensor(rows.class_name.map(CLASSES.index).to_numpy(), device="cuda",
                                dtype=torch.long), len(CLASSES)).float()
    target, overlap = [torch.as_tensor(x, device="cuda") for x in (xy, ov)]
    error = torch.tensor(rows.error.fillna(0).to_numpy(), device="cuda", dtype=torch.float32)
    weight = torch.tensor(np.where(rows.part == "synth", .30, 1.), device="cuda", dtype=torch.float32)
    result = []
    for day in [*DAYS, "test"]:
        fit, pred = train_masks(rows, day)
        indices, predict = [torch.tensor(np.flatnonzero(m), device="cuda") for m in (fit,pred)]
        signals = {}
        for fraction in (0., .35):
            pred_array = np.mean([fit_predict(tokens, augment, lookup, cls, target, overlap, error,
                                              weight, indices, predict, s, fraction)
                                  for s in (117, 312, 614)], axis=0)
            tag = "refiner_clean" if fraction == 0 else "refiner_occluded"
            signals[tag] = score(rows.loc[pred], pred_array)
            print(day, tag, flush=True)
        result.append(rows.loc[pred, ["id"]].assign(**signals))
        pd.concat(result, ignore_index=True).to_csv(CACHE / "refiner.csv", index=False)


if __name__ == "__main__":
    main()
