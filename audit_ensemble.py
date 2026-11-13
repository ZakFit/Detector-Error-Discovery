"""Day-held-out comparisons of error-ranking candidates on the RASCAR case."""
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import BOX, D, FEATURES, build, iou_matrix
from opinion_features import OPINION_FEATURES, opinion_features
from reannotate import agreement, amodal_iou, full_frame_iou, score

ROOT = Path(__file__).resolve().parent
PERSON_COLUMNS = [
    "confidence", "det_iou", "det_score", "bottom_delta", "top_delta",
    "log_area_ratio", "pose_box_iou", "pose_quality", "pose_outside", "feet_missing",
]
COMPACT = [
    "open", "agree", "amodal", "confidence", "full_frame", "seg_iou",
    "pose_iou", "pose_dy2", "wrist_dist", "wrist_conf_in",
]
ONEHOT = [col for col in FEATURES if col.startswith("is_")]


def branch_csv(path):
    ref = "69d957e"
    return pd.read_csv(io.BytesIO(subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)))


def external_features(part):
    df = build(part).reset_index(drop=True)
    df["day"] = df.frame_id.str[-15:-7]
    df["agree"] = agreement(df)
    df["amodal"] = amodal_iou(df)
    df["open"] = score(df)
    df["full_frame"] = full_frame_iou(df)
    return df.merge(opinion_features(df), on="id", validate="one_to_one")


def ft_signals(df, name):
    ref = pd.read_csv(ROOT / "data" / f"ft_{name}.csv")
    ref = ref[ref.ext_conf >= 0.10]
    out = pd.DataFrame(0.0, index=df.index, columns=["iou", "weighted", "dupe", "ref_conf", "cross"])
    for (frame, cls), group in df.groupby(["frame_id", "class_name"]):
        source = ref[ref.frame_id == frame]
        good = source[source.ext_class == cls]
        wrong = source[source.ext_class != cls]
        boxes = group[BOX].to_numpy()
        if len(wrong):
            cross = iou_matrix(boxes, wrong[BOX].to_numpy()) * wrong.ext_conf.to_numpy()
            out.loc[group.index, "cross"] = cross.max(axis=1)
        if good.empty:
            continue
        overlaps = iou_matrix(boxes, good[BOX].to_numpy())
        quality = good.ext_conf.to_numpy()
        top = overlaps.argmax(axis=1)
        best = overlaps.max(axis=1)
        out.loc[group.index, "iou"] = best
        out.loc[group.index, "weighted"] = (overlaps * quality).max(axis=1)
        out.loc[group.index, "ref_conf"] = np.where(best >= 0.25, quality[top], 0.0)
        feasible = overlaps >= 0.6
        assignment_weight = np.where(feasible, len(group) + overlaps, 0)
        rr, cc = linear_sum_assignment(assignment_weight, maximize=True)
        valid = feasible[rr, cc]
        unmatched = np.ones(len(group), dtype=bool)
        unmatched[rr[valid]] = False
        out.loc[group.index, "dupe"] = (unmatched & feasible.any(axis=1)).astype(float)
    return out


def meta_model(train, pred, name):
    base_cols = COMPACT + ONEHOT
    if name == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.08, max_iter=1000))
        return model.fit(train[base_cols].fillna(0), train.error).predict_proba(pred[base_cols].fillna(0))[:, 1]
    cols = {
        "cat_tab": FEATURES,
        "cat_compact": base_cols,
        "cat_full": list(dict.fromkeys(COMPACT + OPINION_FEATURES + FEATURES)),
    }[name]
    model = CatBoostClassifier(
        iterations=180, depth=3, learning_rate=0.035, l2_leaf_reg=8,
        random_seed=61, verbose=False, allow_writing_files=False, thread_count=4,
    )
    return model.fit(train[cols], train.error).predict_proba(pred[cols])[:, 1]


def person_scores(train, pred, features):
    left = train[["id", "error"]].merge(features, on="id")
    right = pred[["id"]].merge(features, on="id")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    model.fit(left[PERSON_COLUMNS].fillna(0), left.error)
    return pd.Series(model.predict_proba(right[PERSON_COLUMNS].fillna(0))[:, 1], index=right.id)


def person_rerank(pred, base, person_score, weight):
    output = np.array(base).copy()
    mask = pred.class_name.to_numpy() == "person"
    visual = pred.loc[mask, "id"].map(person_score).to_numpy()
    order = np.argsort((1 - weight) * rankdata(output[mask]) + weight * rankdata(visual), kind="stable")
    revised = np.empty_like(visual)
    revised[order] = np.sort(output[mask])
    output[mask] = revised
    return output


def embedding_signals(train, pred, cache):
    by_id = pd.Index(cache["ids"])
    left = by_id.get_indexer(train.id)
    right = by_id.get_indexer(pred.id)
    assert min(left.min(), right.min()) >= 0 and by_id.is_unique
    emb = cache["embedding"]
    vec = emb.reshape(len(emb), -1) / np.sqrt(emb.shape[1])
    if not np.isfinite(vec).all():
        raise ValueError("Non-finite embeddings")
    train_days = train.day.unique()
    pool = (cache["split"] == "pool") & np.isin(cache["day"], train_days)
    # Match the same annotated class and shrink sparse classes toward the fold prior.
    nn = np.full(len(pred), train.error.mean())
    for cls, rows in pred.groupby("class_name"):
        pi = pred.index.get_indexer(rows.index)
        mask = train.class_name.to_numpy() == cls
        sim = vec[right[pi]] @ vec[left[mask]].T
        k = min(7, sim.shape[1])
        if k == 0:
            continue
        idx = np.argsort(-sim, axis=1)[:, :k]
        weights = np.exp(12 * (np.take_along_axis(sim, idx, 1) - sim.max(1, keepdims=True)))
        nn[pi] = (weights * train.error.to_numpy()[mask][idx]).sum(1) / weights.sum(1)

    def design(positions):
        onehot = np.column_stack([cache["cls"][positions] == c for c in ("person", "nude_hands", "cup", "tray")])
        return np.column_stack([vec[positions] * 3, onehot, cache["confidence"][positions]])

    pool_indices = np.flatnonzero(pool)
    x, x_pool, x_pred = [design(pos) for pos in (left, pool_indices, right)]
    y = train.error.to_numpy()
    teacher = LogisticRegression(C=2, max_iter=1000).fit(x, y)
    soft = teacher.predict_proba(x_pool)[:, 1]
    pseudo_y = soft >= 0.5
    pseudo = (soft >= 0.8) | (soft <= 0.2)
    # Bound the contribution from any class/label, and give human labels more weight.
    chosen = []
    for cls in np.unique(cache["cls"][pool_indices]):
        for label in (False, True):
            ids = np.flatnonzero(pseudo & (pseudo_y == label) & (cache["cls"][pool_indices] == cls))
            chosen.extend(ids[np.argsort(-np.abs(soft[ids] - 0.5))[:60]])
    chosen = np.array(chosen, dtype=int)
    weights = np.r_[np.ones(len(y)), np.full(len(chosen), 0.25)]
    student = LogisticRegression(C=2, max_iter=1000)
    student.fit(np.vstack([x, x_pool[chosen]]), np.r_[y, pseudo_y[chosen]], sample_weight=weights)
    stats = (int(pool.sum()), int((~pseudo_y[chosen]).sum()), int(pseudo_y[chosen].sum()))
    return nn, teacher.predict_proba(x_pred)[:, 1], student.predict_proba(x_pred)[:, 1], stats


def candidates(train, pred, ft_name, person_features, cache):
    ft = ft_signals(pred, ft_name)
    base = pred.open.to_numpy()
    quality = 1 - ft.iou.to_numpy()
    original = (base + quality) / 2
    options = {
        "confidence": 1 - pred.confidence.to_numpy(),
        "open_only": base,
        "current_0842": original,
        "ft_only": quality,
        "ft_conf_50": (base + 1 - ft.weighted.to_numpy()) / 2,
        "ft_25": 0.75 * base + 0.25 * quality,
        "ft_35": 0.65 * base + 0.35 * quality,
        "ft_65": 0.35 * base + 0.65 * quality,
        "ft_dupe": original + 0.15 * ft.dupe.to_numpy() * (1 - original),
    }
    # Downweight weak positive evidence from a low-confidence YOLO box.
    # Treat the absence of any sufficiently overlapping YOLO box as missing evidence.
    reliability = 0.20 + 0.30 * ft.ref_conf.to_numpy()
    options["ft_reliability"] = (1 - reliability) * base + reliability * quality
    pscore = person_scores(train, pred, person_features)
    for name in ("open_only", "current_0842", "ft_35", "ft_reliability"):
        options[name + "_person"] = person_rerank(pred, options[name], pscore, 0.5)
    for name in ("logistic", "cat_tab", "cat_compact", "cat_full"):
        learned = meta_model(train, pred, name)
        options[name] = learned
        options["current_plus_" + name] = 0.75 * original + 0.25 * learned
    if cache is not None:
        nn, visual, semi, stats = embedding_signals(train, pred, cache)
        print("pool examined / pseudo valid / pseudo error:", stats, flush=True)
        for name, val in (("dino_knn", nn), ("dino_lr", visual), ("dino_semi", semi)):
            options[name] = val
            options["current_plus_" + name] = 0.75 * original + 0.25 * val
            options["ft35_person_plus_" + name] = 0.75 * options["ft_35_person"] + 0.25 * val
    return options, ft


def ranking_metrics(labels, values):
    order = np.argsort(-values, kind="stable")
    return {
        "ap": average_precision_score(labels, values),
        "errors_at_20": labels[order[:20]].sum(),
        "errors_at_50": labels[order[:50]].sum(),
    }


def main():
    train = external_features("train")
    test = external_features("test")
    person = branch_csv("outputs/features.csv")
    person = person.drop(columns=["error"])
    assert len(train) == 198 and len(test) == 362
    assert set(train[train.class_name == "person"].id) | set(test[test.class_name == "person"].id) == set(person.id)
    cache_path = D.parent / "audit_dino.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else None
    print("embeddings:", cache is not None, flush=True)
    records, folds = [], []
    for fold, day in (("d26", "20261025"), ("d25", "20261026")):
        pred = train[train.day == day].copy()
        print(day, len(pred), "errors", int(pred.error.sum()), flush=True)
        options, ft = candidates(train[train.day != day], pred, fold, person, cache)
        print("duplicates at IoU 0.6:", int(ft.dupe.sum()), "errors:", int(pred.loc[ft.dupe == 1, "error"].sum()), flush=True)
        for name, values in options.items():
            for cls in ["all"] + sorted(pred.class_name.unique()):
                mask = np.ones(len(pred), dtype=bool) if cls == "all" else pred.class_name.to_numpy() == cls
                labels = pred.error.to_numpy()[mask]
                records.append({"name": name, "day": day, "class": cls, "n": len(labels),
                                **ranking_metrics(labels, np.asarray(values)[mask])})
        folds.append(pred[["id", "frame_id", "day", "class_name", "error"]].assign(**options))
    results = pd.DataFrame(records)
    table = results[results["class"] == "all"].pivot(index="name", columns="day", values=["ap", "errors_at_20", "errors_at_50"])
    table["mean_ap"] = table["ap"].mean(axis=1)
    print(table.sort_values("mean_ap", ascending=False).round(4).to_string(), flush=True)
    results.to_csv(D.parent / "audit_metrics.csv", index=False)
    pd.concat(folds).to_csv(D.parent / "audit_oof.csv", index=False)

    options, ft = candidates(train, test, "all", person, cache)
    print("test duplicate penalty candidates:", int(ft.dupe.sum()), flush=True)
    actual = pd.read_csv(ROOT / "submissions" / "sub_ft_all.csv").set_index("id").score
    reproduction = pd.Series(options["current_0842"], index=test.id)
    np.testing.assert_allclose(actual, reproduction.reindex(actual.index), rtol=0, atol=1e-12)
    print("Current submission reproduced over", len(actual), "ids", flush=True)
    pd.DataFrame({"id": test.id, **options}).to_csv(D.parent / "audit_test_scores.csv", index=False)


if __name__ == "__main__":
    main()
