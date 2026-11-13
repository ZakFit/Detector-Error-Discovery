"""Train full-frame and crop specialists on hand annotations, keeping days separate."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "data" / "gpu_runs"
os.environ["YOLO_CONFIG_DIR"] = str(WORK / "config")
os.environ["YOLO_OFFLINE"] = "true"
sys.path.insert(0, str(ROOT.parent / "gpu_deps"))

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO, settings

from features import BOX, CLASSES, D, load

cv2.setNumThreads(0)


def targets():
    annotations = pd.read_csv(D / "human_boxes_train.csv")
    aliases = json.loads((D / "scoring_policy.json").read_text())["aliases"]
    annotations["class_name"] = annotations.class_name.replace(aliases)
    annotations = annotations[annotations.class_name.isin(CLASSES)].copy()
    for key, col, scale, delta in (
        ("x1", "cx", 1280, -annotations.w / 2), ("x2", "cx", 1280, annotations.w / 2),
        ("y1", "cy", 720, -annotations.h / 2), ("y2", "cy", 720, annotations.h / 2),
    ):
        annotations[key] = (annotations[col] + delta) * scale
    return annotations


def window(box, ratio):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max(55, (x2 - x1) * ratio / 2), max(55, (y2 - y1) * ratio / 2)
    return tuple(np.round([max(0, cx - hw), max(0, cy - hh), min(1280, cx + hw), min(720, cy + hh)]).astype(int))


def write_labels(path, annot, crop):
    x1, y1, x2, y2 = crop
    w, h = x2 - x1, y2 - y1
    boxes = annot[BOX].to_numpy()
    clip = np.clip(boxes, [x1, y1, x1, y1], [x2, y2, x2, y2])
    wh = clip[:, 2:] - clip[:, :2]
    keep = (wh.min(axis=1) >= 2) & (wh.prod(axis=1) >= 0.60 * (boxes[:, 2:] - boxes[:, :2]).prod(axis=1))
    cxcy = ((clip[:, :2] + clip[:, 2:]) / 2 - [x1, y1]) / [w, h]
    norm = np.column_stack([cxcy, wh / [w, h]])
    lines = [f"{CLASSES.index(cls)} " + " ".join(f"{v:.7f}" for v in coords)
             for cls, coords, valid in zip(annot.class_name, norm, keep) if valid]
    path.write_text("\n".join(lines), encoding="utf-8")


def dataset(mode, fold, frames, annotations, detections):
    directory = WORK / f"dataset_{mode}_{fold}"
    images, labels = directory / "images", directory / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(exist_ok=True)
    for frame in frames:
        annot = annotations[annotations.frame_id == frame]
        proposals = detections[detections.frame_id == frame]
        if mode == "full":
            dst = images / f"{frame}.jpg"
            if not dst.exists():
                shutil.copyfile(D / dst.name, dst)
            write_labels(labels / f"{frame}.txt", annot, (0, 0, 1280, 720))
            continue
        img = cv2.imread(str(D / f"{frame}.jpg"))
        boxes = pd.concat([annot, proposals])
        boxes = boxes[boxes.class_name != "person"]
        for i, box in enumerate(boxes[BOX].to_numpy()):
            x1, y1, x2, y2 = crop = window(box, 4)
            cv2.imwrite(str(images / f"{frame}_{i}.jpg"), img[y1:y2, x1:x2])
            write_labels(labels / f"{frame}_{i}.txt", annot, crop)
    cfg = directory / "dataset.yaml"
    cfg.write_text(f"path: {directory.as_posix()}\ntrain: images\nval: images\nnames: {dict(enumerate(CLASSES))}\n",
                   encoding="utf-8")
    return cfg


def train(mode, fold, cfg, epochs):
    run = WORK / f"{mode}_{fold}"
    checkpoint = run / "weights" / "last.pt"
    if checkpoint.exists() and (run / "finished").exists():
        return YOLO(str(checkpoint))
    model = YOLO(str(ROOT / "weights" / "yolo11s.pt"))
    model.train(data=str(cfg), epochs=epochs, imgsz=960 if mode == "full" else 384,
                batch=4 if mode == "full" else 12, device=0, workers=0,
                amp=os.environ.get("RASCAR_YOLO_AMP", "bf16"),
                project=str(WORK), name=run.name, exist_ok=True, val=False, plots=False,
                verbose=False, save=True, seed=71, cache=False, close_mosaic=8,
                mosaic=0.6, scale=0.35, patience=0)
    (run / "finished").touch()
    return YOLO(str(checkpoint))


def infer_full(model, frames, outfile):
    rows = []
    for res in model.predict([str(D / f"{f}.jpg") for f in frames], stream=True,
                             batch=4, device=0, imgsz=960, conf=0.05, verbose=False):
        for x1, y1, x2, y2, confidence, cls in res.boxes.data.cpu().tolist():
            rows.append((Path(res.path).stem, CLASSES[int(cls)], confidence, x1, y1, x2, y2))
    pd.DataFrame(rows, columns=["frame_id", "ext_class", "ext_conf", *BOX]).to_csv(outfile, index=False)


def infer_crops(model, detections, outfile):
    rows = []
    subset = detections[detections.class_name != "person"]
    for frame, group in subset.groupby("frame_id"):
        img = cv2.imread(str(D / f"{frame}.jpg"))
        windows = [window(box, 4) for box in group[BOX].to_numpy()]
        crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
        pred = model.predict(crops, device=0, imgsz=384, conf=0.05, verbose=False)
        for ident, bounds, res in zip(group.id, windows, pred):
            x, y, _, _ = bounds
            for x1, y1, x2, y2, confidence, cls in res.boxes.data.cpu().tolist():
                rows.append((ident, CLASSES[int(cls)], confidence, x1 + x, y1 + y, x2 + x, y2 + y))
    pd.DataFrame(rows, columns=["id", "ext_class", "ext_conf", *BOX]).to_csv(outfile, index=False)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("mode", choices=["full", "crop", "zoom"])
    cli.add_argument("folds", nargs="+", choices=["d25", "d26", "all"])
    cli.add_argument("--epochs", type=int, default=36)
    args = cli.parse_args()
    settings.update({"sync": False})
    annotations = targets()
    tr, _, train_frames, _ = load("train")
    te, _, test_frames, _ = load("test")
    tr, te = [p[p.class_name.isin(CLASSES)] for p in (tr, te)]
    for fold in args.folds:
        out = WORK / f"{args.mode}_{fold}.csv"
        if out.exists():
            continue
        if fold == "all":
            fit_frames, eval_frames, eval_det = train_frames.frame_id, test_frames.frame_id, te
        else:
            mask = train_frames.frame_id.str[-9:-7] == fold[-2:]
            fit_frames, eval_frames = train_frames.frame_id[mask], train_frames.frame_id[~mask]
            eval_det = tr[tr.frame_id.isin(eval_frames)]
        if args.mode == "zoom":
            checkpoint = WORK / f"full_{fold}" / "weights" / "last.pt"
            model = YOLO(str(checkpoint))
        else:
            cfg = dataset(args.mode, fold, fit_frames, annotations, tr)
            model = train(args.mode, fold, cfg, args.epochs)
        if args.mode == "full":
            infer_full(model, eval_frames, out)
        else:
            infer_crops(model, eval_det, out)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
