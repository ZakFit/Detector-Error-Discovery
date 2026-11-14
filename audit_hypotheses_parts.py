"""Distinguish full person boxes from detections covering body parts."""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from audit_hypotheses_features import CACHE, metadata
from audit_hypotheses_scene import DAYS
from audit_hypotheses_visual import train_masks
from features import BOX, iou_matrix


def relation(rows, gt):
    coverage = np.zeros((len(rows), 2))
    people = gt[gt.class_name == "person"]
    parts = gt[gt.class_name.isin(["uniform_not_ok", "uniform_ok", "hair"])]
    for frame, group in rows.groupby("frame_id"):
        boxes = group[BOX].to_numpy()
        for j, known in enumerate((people, parts)):
            coverage[group.index, j] = iou_matrix(boxes, known.loc[known.frame_id == frame, BOX].to_numpy()).max(1, initial=0)
    return coverage


def fit_model(rows, x, fit, predict, coverage, supervised):
    person, part = coverage.T
    annotated = rows.part == "gt"
    full = annotated & (rows.class_name == "person")
    clipped = annotated & rows.class_name.isin(["uniform_not_ok", "uniform_ok", "hair"]) & (person < .58) & (person > .10)
    real = (rows.part == "train") & (rows.class_name == "person")
    label = full | (real & (rows.error == 0))
    eligible = full | clipped
    if supervised:
        eligible |= real & ((rows.error == 0) | ((person >= .15) & (person < .60)))
    chosen = (eligible & fit).to_numpy()
    fit = LogisticRegression(C=8, max_iter=500, class_weight="balanced")
    fit.fit(x[chosen], label[chosen], sample_weight=np.where(real[chosen], 1.3, 1))
    return -fit.decision_function(x[predict])


def main():
    _, gt, _ = metadata()
    rows = pd.read_csv(CACHE / "rows.csv")
    select = ~rows.occluded & (rows.view == 0)
    tight = torch.load(CACHE / "tight.pt", weights_only=True).numpy()[select]
    tokens = torch.load(CACHE / "context.pt", weights_only=True)[np.flatnonzero(select)]
    global_context = tokens[:, -100:].float().mean(1).numpy()
    rows = rows[select].reset_index(drop=True)
    box = rows[BOX].to_numpy()
    wh = np.maximum(4, box[:, 2:] - box[:, :2])
    geom = .18 * np.column_stack([np.log(wh[:, 0] / wh[:, 1]), np.log(wh.prod(1)/10000)])
    x = np.column_stack([normalize(tight.astype(np.float32)), .7 * normalize(global_context), geom])
    cov = relation(rows, gt)
    results = []
    for day in [*DAYS, "test"]:
        fit, predict = train_masks(rows, day)
        results.append(rows.loc[predict, ["id"]].assign(
            parts_gt=fit_model(rows, x, fit, predict, cov, False),
            parts_supervised=fit_model(rows, x, fit, predict, cov, True),
        ))
    pd.concat(results, ignore_index=True).to_csv(CACHE / "parts.csv", index=False)


if __name__ == "__main__":
    main()
