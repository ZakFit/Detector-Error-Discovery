"""Compare a frozen DINOv2 crop encoder with task-adapted final blocks."""
import argparse
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.nn import functional as F

from audit_gpu_yolo import targets
from features import BOX, CLASSES, D, iou_matrix, load

SIZE = 140
BATCH = 32
WORK = D.parent / "gpu_runs"


def candidates(gt):
    rng = np.random.default_rng(213)
    records = []
    for (frame, cls), group in gt.groupby(["frame_id", "class_name"]):
        boxes = group[BOX].to_numpy()
        for box in boxes:
            wh = box[2:] - box[:2]
            center = (box[:2] + box[2:]) / 2
            proposed = [box]
            for _ in range(12):
                scale = rng.choice([0.10, 0.23, 0.42])
                c = center + rng.normal(0, scale, 2) * wh
                size = wh * np.exp(rng.normal(0, scale, 2))
                proposed.append(np.r_[c - size / 2, c + size / 2])
            proposed = np.clip(proposed, [0, 0, 0, 0], [1280, 720, 1280, 720])
            scores = iou_matrix(proposed, boxes).max(axis=1)
            for proposal, quality in zip(proposed, scores):
                if min(proposal[2:] - proposal[:2]) < 4:
                    continue
                records.append((frame, cls, *proposal, float(quality < 0.60)))
    return pd.DataFrame(records, columns=["frame_id", "class_name", *BOX, "error"])


def crops(rows):
    cache = {}
    for row in rows.itertuples():
        if row.frame_id not in cache:
            cache[row.frame_id] = cv2.imread(str(D / f"{row.frame_id}.jpg"))
        image = cache[row.frame_id]
        x1, y1, x2, y2 = (getattr(row, col) for col in BOX)
        # The candidate occupies a consistent central region and the encoder sees beyond its edges.
        dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
        x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
        x2, y2 = min(image.shape[1], int(np.ceil(x2 + dx))), min(image.shape[0], int(np.ceil(y2 + dy)))
        tile = cv2.resize(image[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        yield cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def extract(model, rows):
    output = []
    images = []
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, image in enumerate(crops(rows), 1):
            images.append(image)
            if len(images) != BATCH and i != len(rows):
                continue
            x = torch.from_numpy(np.stack(images)).cuda().permute(0, 3, 1, 2) / 255
            x = model.norm_pre(model._pos_embed(model.patch_embed((x - mean) / std)))
            x = model.blocks[:-2](x)
            output.append(x.to(torch.float16).cpu())
            images.clear()
    return torch.cat(output)


class CropHead(nn.Module):
    def __init__(self, backbone, adapt):
        super().__init__()
        import copy
        self.blocks = copy.deepcopy(backbone.blocks[-2:]).requires_grad_(adapt)
        self.norm = copy.deepcopy(backbone.norm).requires_grad_(adapt)
        self.head = nn.Sequential(nn.LayerNorm(1024 + len(CLASSES)), nn.Linear(1024 + len(CLASSES), 128),
                                  nn.GELU(), nn.Dropout(0.20), nn.Linear(128, 1))

    def forward(self, tokens, cls):
        vec = self.norm(self.blocks(tokens.float()))[:, 0]
        return self.head(torch.cat([vec, cls], dim=1)).squeeze(1)


def fit(backbone, tokens, cls, rows, tr, te, adapt, checkpoint=None):
    torch.manual_seed(117)
    model = CropHead(backbone, adapt).cuda()
    params = [{"params": model.head.parameters(), "lr": 3e-4}]
    if adapt:
        params.append({"params": list(model.blocks.parameters()) + list(model.norm.parameters()), "lr": 1e-5})
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    labels = torch.tensor(rows.error.fillna(0).to_numpy(), dtype=torch.float32)
    weights = torch.tensor(np.where(rows.part == "synth", 0.25, 1.0), dtype=torch.float32)
    generator = torch.Generator().manual_seed(312)
    for epoch in range(8):
        model.train()
        order = tr[torch.randperm(len(tr), generator=generator)]
        losses = []
        for batch in order.split(BATCH):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens[batch].cuda(), cls[batch].cuda())
                loss = F.binary_cross_entropy_with_logits(logits, labels[batch].cuda(), weight=weights[batch].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.detach())
        print("adapt", adapt, "epoch", epoch + 1, "loss", float(torch.stack(losses).mean()), flush=True)
    model.eval()
    if checkpoint is not None:
        torch.save(model.state_dict(), checkpoint)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.cat([
            model(tokens[b].cuda(), cls[b].cuda()).float().sigmoid().cpu() for b in te.split(BATCH)
        ]).numpy()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("folds", nargs="+", choices=["d25", "d26", "all"])
    args = cli.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    tr, *_ = load("train")
    te, *_ = load("test")
    syn = candidates(targets())
    rows = pd.concat([tr.assign(part="train"), te.assign(part="test"), syn.assign(part="synth")], ignore_index=True)
    rows = rows[rows.class_name.isin(CLASSES)].reset_index(drop=True)
    days = rows.frame_id.str[-9:-7]
    cls = F.one_hot(torch.tensor(rows.class_name.map(CLASSES.index).to_numpy()), len(CLASSES)).float()
    WORK.mkdir(parents=True, exist_ok=True)
    ckpt = WORK / "dino_tokens.pt"
    backbone = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=SIZE)
    if not ckpt.exists():
        print("extracting frozen tokens:", len(rows), flush=True)
        torch.save(extract(backbone.eval().cuda(), rows), ckpt)
        backbone.cpu()
        torch.cuda.empty_cache()
    tokens = torch.load(ckpt, weights_only=True)
    assert len(tokens) == len(rows)
    for fold in args.folds:
        out = WORK / f"dino_{fold}.csv"
        if out.exists():
            continue
        fit_mask = rows.part != "test"
        if fold == "all":
            predict_mask = rows.part == "test"
        else:
            fit_mask &= days == fold[-2:]
            predict_mask = (rows.part == "train") & (days != fold[-2:])
        train_indices = torch.from_numpy(np.flatnonzero(fit_mask))
        test_indices = torch.from_numpy(np.flatnonzero(predict_mask))
        result = rows.loc[predict_mask, ["id"]].copy()
        print(fold, len(train_indices), len(test_indices), flush=True)
        for adapt in (False, True):
            mode = "adapted" if adapt else "frozen"
            checkpoint = WORK / f"dino_{fold}_{mode}.pt" if fold == "all" else None
            result[mode] = fit(backbone, tokens, cls, rows, train_indices, test_indices, adapt, checkpoint)
        result.to_csv(out, index=False)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
