"""Cache frozen DINOv2 embeddings for target crops and a sampled machine pool."""
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import BOX, CLASSES, D, load


def main():
    out = D.parent / "audit_dino.npz"
    target = []
    excluded = set()
    for split in ("train", "test"):
        det, _, ctx, _ = load(split)
        target.append(det[det.class_name.isin(CLASSES)].assign(split=split))
        excluded.update(ctx[["frame_id", "previous", "next"]].stack())
    pool = pd.read_csv(D / "machine_predictions.csv").rename(columns={"detection_id": "id"})
    pool["day"] = pool.frame_id.str[-15:-7]
    pool = pool[pool.class_name.isin(CLASSES) & ~pool.frame_id.isin(excluded)]
    pool = pd.concat([
        group.sample(min(240, len(group)), random_state=91)
        for _, group in pool.groupby(["day", "class_name"])
    ]).assign(split="pool")
    rows = pd.concat(target + [pool], ignore_index=True)
    rows["day"] = rows.frame_id.str[-15:-7]
    print(rows.groupby(["split", "day", "class_name"]).size().to_string(), flush=True)

    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224,
    ).eval().cuda()
    prep = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    embeddings = []
    batch = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i, row in enumerate(rows.itertuples(), 1):
            with Image.open(D / f"{row.frame_id}.jpg") as image:
                x1, y1, x2, y2 = (getattr(row, col) for col in BOX)
                sx, sy = image.width / 1280, image.height / 720
                dx, dy = x2 - x1, y2 - y1
                for margin in (0.05, 0.30):
                    rect = (max(0, round((x1 - margin * dx) * sx)),
                            max(0, round((y1 - margin * dy) * sy)),
                            min(image.width, round((x2 + margin * dx) * sx)),
                            min(image.height, round((y2 + margin * dy) * sy)))
                    batch.append(prep(image.crop(rect).convert("RGB")))
            if len(batch) == 32 or i == len(rows):
                e = model(torch.stack(batch).cuda()).float()
                embeddings.append(torch.nn.functional.normalize(e).cpu().numpy())
                batch.clear()
            if i % 320 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    embeddings = np.concatenate(embeddings).reshape(len(rows), 2, -1)
    np.savez_compressed(
        out, embedding=embeddings, ids=rows.id.to_numpy(dtype=str),
        frame=rows.frame_id.to_numpy(dtype=str), split=rows.split.to_numpy(dtype=str),
        day=rows.day.to_numpy(dtype=str), cls=rows.class_name.to_numpy(dtype=str),
        confidence=rows.confidence.to_numpy(),
    )
    print(out, embeddings.shape, flush=True)


if __name__ == "__main__":
    main()
