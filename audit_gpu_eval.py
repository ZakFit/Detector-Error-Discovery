"""Evaluate GPU candidates by held-out day and export a chosen test candidate."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from audit_ensemble import branch_csv, external_features, person_rerank, person_scores
from features import BOX, D, iou_matrix

WORK = D.parent / "gpu_runs"
BASE = "ft_reliability_person"


def yolo_signal(det, path, crop):
    if not path.exists():
        return None
    reference = pd.read_csv(path)
    ref = reference[reference.ext_conf >= 0.1]
    key = "id" if crop else "frame_id"
    groups = {(ident, cls): rows for (ident, cls), rows in ref.groupby([key, "ext_class"])}
    overlaps, quality = np.zeros(len(det)), np.zeros(len(det))
    for i, row in enumerate(det.itertuples()):
        group = groups.get((getattr(row, key), row.class_name))
        if group is None:
            continue
        box = np.array([[getattr(row, b) for b in BOX]])
        ious = iou_matrix(box, group[BOX].to_numpy())[0]
        overlaps[i] = ious.max()
        quality[i] = float(group.ext_conf.iloc[ious.argmax()]) if overlaps[i] >= 0.25 else 0
    return overlaps, quality


def variants(fold, det, base, train, person, current=None):
    small = det.class_name.to_numpy() != "person"
    result = {"previous": base.copy()}
    if current is not None:
        result["current_0842"] = current
    full = yolo_signal(det, WORK / f"full_{fold}.csv", crop=False)
    crop = yolo_signal(det, WORK / f"crop_{fold}.csv", crop=True)
    pscore = person_scores(train, det, person)
    if full is not None:
        iou, confidence = full
        weight = 0.20 + 0.30 * confidence
        score = (1 - weight) * det.open.to_numpy() + weight * (1 - iou)
        result["full_yolo11s"] = person_rerank(det, score, pscore, 0.5)
        result["mean_yolo11n_s"] = 0.5 * (base + result["full_yolo11s"])
        if current is not None:
            result["current_full"] = (1 - weight) * current + weight * (1 - iou)
    if crop is not None:
        iou, confidence = crop
        result["crop_yolo11s"] = np.where(small, 0.75 * base + 0.25 * (1 - iou), base)
        weight = 0.12 + 0.25 * confidence
        result["crop_reliability"] = np.where(small, (1 - weight) * base + weight * (1 - iou), base)
        hands = det.class_name.to_numpy() == "nude_hands"
        result["crop_hands"] = np.where(hands, result["crop_reliability"], base)
        if current is not None:
            result["current_crop"] = np.where(small, (1 - weight) * current + weight * (1 - iou), current)
        if full is not None:
            mixture = result["mean_yolo11n_s"]
            result["full_crop"] = np.where(small, (1 - weight) * mixture + weight * (1 - iou), mixture)
    zoom = yolo_signal(det, WORK / f"zoom_{fold}.csv", crop=True)
    if zoom is not None:
        iou, confidence = zoom
        weight = 0.12 + 0.25 * confidence
        result["zoom_full_weights"] = np.where(small, (1 - weight) * base + weight * (1 - iou), base)
    dino = WORK / f"dino_{fold}.csv"
    if dino.exists():
        dino = pd.read_csv(dino).set_index("id")
        for mode in ("frozen", "adapted"):
            score = det.id.map(dino[mode]).to_numpy()
            assert np.isfinite(score).all()
            result[f"dino_{mode}"] = score
            result[f"previous_dino_{mode}"] = 0.80 * base + 0.20 * score
            if current is not None:
                result[f"current_dino_{mode}"] = 0.80 * current + 0.20 * score
            if full is not None and crop is not None:
                result[f"full_crop_dino_{mode}"] = 0.80 * result["full_crop"] + 0.20 * score
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--export", help="candidate name")
    args = cli.parse_args()
    tr = external_features("train")
    oof = pd.read_csv(D.parent / "audit_oof.csv").set_index("id")
    person = branch_csv("outputs/features.csv").drop(columns=["error"])
    records, predictions = [], []
    for fold, day in (("d26", "20261025"), ("d25", "20261026")):
        det = tr[tr.day == day].reset_index(drop=True)
        base = det.id.map(oof[BASE]).to_numpy()
        current = det.id.map(oof["current_0842"]).to_numpy()
        options = variants(fold, det, base, tr[tr.day != day], person, current)
        predictions.append(det[["id", "day", "frame_id", "class_name", "error"]].assign(**options))
        for name, values in options.items():
            for cls in ["all"] + sorted(det.class_name.unique()):
                mask = np.ones(len(det), dtype=bool) if cls == "all" else det.class_name.to_numpy() == cls
                y, s = det.error.to_numpy()[mask], values[mask]
                order = np.argsort(-s, kind="stable")
                records.append((name, day, cls, len(y), average_precision_score(y, s),
                                int(y[order[:20]].sum()), int(y[order[:50]].sum())))
    metrics = pd.DataFrame(records, columns=["name", "day", "class", "n", "ap", "errors_at_20", "errors_at_50"])
    pivot = metrics[metrics["class"] == "all"].pivot(index="name", columns="day", values="ap")
    pivot["mean_ap"] = pivot.mean(axis=1, skipna=False)
    print(pivot.sort_values("mean_ap", ascending=False).round(5).to_string())
    metrics.to_csv(WORK / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(WORK / "validation_predictions.csv", index=False)
    if args.export:
        assert args.export != "previous" and pivot.loc[args.export].notna().all()
        te = external_features("test")
        earlier = pd.read_csv(D.parent / "audit_test_scores.csv").set_index("id")
        base = te.id.map(earlier[BASE]).to_numpy()
        current = te.id.map(earlier["current_0842"]).to_numpy()
        vals = variants("all", te, base, tr, person, current)[args.export]
        template = pd.read_csv(D / "sample_submission.csv")[["id"]]
        out = template.merge(pd.DataFrame({"id": te.id, "score": vals}), on="id", validate="one_to_one", how="left")
        assert len(out) == len(vals) == 362 and np.isfinite(out.score).all() and out.score.between(0, 1).all()
        path = D.parent.parent.parent.parent / "outputs" / f"submission_gpu_{args.export}.csv"
        out.to_csv(path, index=False)
        reread = pd.read_csv(path)
        assert reread.id.equals(template.id)
        np.testing.assert_allclose(reread.score, out.score, rtol=0, atol=1e-15)
        print("verified:", path)


if __name__ == "__main__":
    main()
