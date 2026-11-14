"""Day-held-out diagnostics and scene assignment with uncertain boxes."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from sklearn.metrics import average_precision_score
from audit_hypotheses_features import CACHE, metadata
from audit_scene_matching import scene_matching
from audit_uncertainty import OUT, WORK
from features import BOX, D, iou_matrix

BASE_FILE = OUT / "submission_refinement_person_geometry_half.csv"
BASE_COL = "person_geometry_half"
DAYS = ["20261025", "20261026"]


def baseline():
    return pd.read_csv(WORK / "box_refinement_oof.csv", dtype={"day": str}).rename(columns={BASE_COL: "base"})


def match(boxes, truth):
    matrix = iou_matrix(boxes, truth)
    valid = matrix >= .60
    r, c = linear_sum_assignment(np.where(valid, len(boxes) + 1 + matrix, 0), maximize=True)
    result = np.zeros(len(boxes), dtype=bool)
    result[r[valid[r, c]]] = True
    return result


def diagnosis(det, gt):
    result = []
    for (f, cls), source in det.groupby(["frame_id", "class_name"]):
        known = gt[(gt.frame_id == f) & (gt.class_name != "table")]
        bb = source[BOX].to_numpy()
        same = known[known.class_name == cls][BOX].to_numpy()
        other = known[known.class_name != cls][BOX].to_numpy()
        overlap = iou_matrix(bb, same).max(axis=1, initial=0)
        different = iou_matrix(bb, other).max(axis=1, initial=0)
        duplicate = overlap >= .6
        kind = np.select([duplicate, (different >= .50) & (different > overlap), overlap >= .15],
                         ["duplicate", "other_class", "localization"], default="background_or_low_iou")
        correct = source.error.to_numpy() == 0
        kind[correct] = "correct"
        tables = gt[(gt.frame_id == f) & (gt.class_name == "table")][BOX].to_numpy()
        wh = bb[:, 2:] - bb[:, :2]
        intersection = np.maximum(0, np.minimum(bb[:, None, 2:], tables[:, 2:]) -
                                  np.maximum(bb[:, None, :2], tables[:, :2])).prod(2)
        table_overlap = (intersection / np.maximum(wh.prod(1)[:, None], 1)).max(axis=1, initial=0)
        result.append(source[["id"]].assign(kind=kind, max_same_iou=overlap,
                                            table_overlap=table_overlap, matches=match(bb, same)))
    return pd.concat(result, ignore_index=True)


def nms_boxes(ref):
    chosen = []
    boxes = ref[BOX].to_numpy()
    conf = ref.confidence.to_numpy()
    order = np.argsort(-conf, kind="stable")
    while len(order):
        idx = order[0]
        chosen.append(idx)
        order = order[1:]
        if len(order):
            order = order[iou_matrix(boxes[idx:idx+1], boxes[order])[0] < .6]
    return boxes[chosen], conf[chosen]


def sampled_assignment(det, ref, draws=120):
    # Jointly sample the same view for all detections in a frame. Perturb object boxes
    # once per draw so duplicates actually compete for the same object.
    rng = np.random.default_rng(624)
    joined = ref[ref.confidence >= .10].merge(det[["id", "frame_id"]], on="id", validate="many_to_one")
    views = {k: nms_boxes(q) for k, q in joined.groupby(["frame_id", "class_name", "view"])}
    result = np.zeros((len(det), 2))
    for (frame, cls), source in det.groupby(["frame_id", "class_name"]):
        bb = source[BOX].to_numpy()
        for view in np.tile(np.arange(6), draws // 6):
            item = views.get((frame, cls, view))
            if item is None:
                continue
            boxes, conf = item
            wh = boxes[:, 2:] - boxes[:, :2]
            middle = (boxes[:, :2] + boxes[:, 2:]) / 2 + rng.normal(0, .055, wh.shape) * wh
            half = wh / 2 * np.exp(rng.normal(0, .09, wh.shape))
            proposed = np.column_stack([middle-half, middle+half]).clip([0,0,0,0], [1280,720,1280,720])
            keep = rng.random(len(conf)) < expit((conf - .15) / .10)
            result[source.index, 0] += match(bb, proposed) / draws
            result[source.index, 1] += match(bb, proposed[keep]) / draws
    return 1 - result


def main():
    det, gt, _ = metadata()
    fit = det[det.part == "train"]
    labels = diagnosis(fit, gt)
    checked = baseline().merge(labels, on="id", validate="one_to_one")
    checked["agrees_with_label"] = checked.matches == (checked.error == 0)
    print("Matching agrees with labels", int(checked.agrees_with_label.sum()), "/", len(checked))
    print(pd.crosstab(checked.class_name, checked.kind).to_string())
    oracle = []
    for day, g in checked.groupby("day"):
        ap = average_precision_score(g.error, g.base)
        for (cls, kind), rows in g[g.error == 1].groupby(["class_name", "kind"]):
            lift = average_precision_score(g.error, np.where(g.index.isin(rows.index), 2, g.base)) - ap
            oracle.append(dict(day=day, class_name=cls, kind=kind, errors=len(rows), ap_lift=lift))
    print(pd.DataFrame(oracle).sort_values(["day", "ap_lift"], ascending=[True, False]).to_string(index=False))
    checked.to_csv(CACHE / "diagnosis.csv", index=False)
    pd.DataFrame(oracle).to_csv(CACHE / "oracle.csv", index=False)
    records = []
    for day, fold in zip([*DAYS, "test"], ["d26", "d25", "all"]):
        rows = det[det.part == "test" if day == "test" else
                   (det.part == "train") & (det.frame_id.str[-15:-7] == day)].reset_index(drop=True)
        ref = pd.read_csv(WORK / f"uncertainty_{fold}.csv")
        deterministic = scene_matching(rows, ref)
        uncertain, with_absence = sampled_assignment(rows, ref).T
        records.append(rows[["id"]].assign(scene_control=deterministic, scene_uncertain=uncertain,
                                           scene_absence=with_absence))
    pd.concat(records, ignore_index=True).to_csv(CACHE / "scene.csv", index=False)


if __name__ == "__main__":
    main()
