"""Compare hypothesis variants against the 0.90734 Public submission."""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from audit_hypotheses_features import CACHE, metadata
from audit_hypotheses_scene import BASE_FILE, DAYS, baseline
from audit_uncertainty import OUT, rank_mix
from features import D

BRANCHES = {
    "parts_gt": ("person",),
    "parts_supervised": ("person",),
    "refiner_clean": ("person",),
    "refiner_occluded": ("person",),
    "amodal_control": ("person",),
    "amodal_aug": ("person",),
    "semantic": ("small", "person", "all"),
    "critic_control": ("all", "person", "small"),
    "critic_pair": ("all", "person", "small"),
    "scene_control": ("all", "person", "small"),
    "scene_uncertain": ("all", "person", "small"),
    "scene_absence": ("all", "person", "small"),
    "ssl_frozen_probe": ("all",),
    "ssl_adapted_probe": ("all",),
    "ssl_frozen_critic": ("all", "person", "small"),
    "ssl_adapted_critic": ("all", "person", "small"),
}


def rank_group(rows, signal, group, weight=.25):
    base = rows.base.to_numpy()
    x = rows[signal].to_numpy()
    if group == "all":
        return rank_mix(rows, rank_mix(rows, base, x, weight), x, weight, True)
    return rank_mix(rows, base, x, weight, group == "person")


def choices(rows):
    scores = {"base": rows.base.to_numpy()}
    for signal, groups in BRANCHES.items():
        if signal not in rows or rows[signal].isna().any():
            continue
        for group in groups:
            for weight in (.25, .50):
                scores[f"{signal}:{group}:{weight}"] = rank_group(rows, signal, group, weight)
    return scores


def bootstrap(df, names, repetitions=2000):
    rng = np.random.default_rng(715)
    n = len(df)
    frames = [np.asarray(j) for j in df.groupby("frame_id").indices.values()]
    y = df.error.to_numpy()
    # Compute AP with tie groups for each resample using integer row multiplicities.
    samples = rng.integers(0, len(frames), size=(repetitions, len(frames)))
    weights = np.stack([np.bincount(np.concatenate([frames[k] for k in draw]), minlength=n)
                        for draw in samples])
    aps = {}
    for name in names:
        score = df[name].to_numpy()
        order = np.argsort(-score, kind="stable")
        end = np.flatnonzero(np.r_[score[order][1:] != score[order][:-1], True])
        hits = np.cumsum(weights[:, order] * y[order], axis=1)[:, end]
        total = np.cumsum(weights[:, order], axis=1)[:, end]
        precision = hits / np.maximum(total, 1)
        aps[name] = (precision * np.diff(hits, axis=1, prepend=0)).sum(1) / np.maximum(hits[:, -1], 1)
    return {name: np.quantile(aps[name] - aps["base"], [.025, .975]) for name in names if name != "base"}


def joined():
    det, _, _ = metadata()
    test = det[det.part == "test"][["id", "class_name", "frame_id"]].assign(day="test")
    test = test.merge(pd.read_csv(BASE_FILE).rename(columns={"score": "base"}), on="id", validate="one_to_one")
    source = pd.concat([baseline()[["id", "frame_id", "class_name", "error", "day", "base"]], test],
                       ignore_index=True)
    for name in ("scene", "visual", "ssl", "refiner", "parts"):
        path = CACHE / f"{name}.csv"
        if path.exists():
            source = source.merge(pd.read_csv(path).drop(columns="fold", errors="ignore"),
                                  on="id", how="left", validate="one_to_one")
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args()
    data = joined()
    metrics, predictions = [], []
    for day in DAYS:
        rows = data[data.day == day].reset_index(drop=True)
        scores = choices(rows)
        all_names = [name for name in BRANCHES if name in rows and rows[name].notna().all()]
        for name in all_names:
            for group in ["all", "person", "small"]:
                keep = np.ones(len(rows), dtype=bool) if group == "all" else (
                    rows.class_name == "person" if group == "person" else rows.class_name != "person")
                metrics.append(dict(day=day, name=name + ":" + group, mode="standalone",
                                    ap=average_precision_score(rows.error[keep], rows[name][keep])))
        for name, score in scores.items():
            metrics.append(dict(day=day, name=name, mode="blend", ap=average_precision_score(rows.error, score)))
        pred = pd.concat([rows[["id", "frame_id", "class_name", "day", "error"]], pd.DataFrame(scores)], axis=1)
        for name in args.candidate:
            if "+" in name and all(part in pred for part in name.split("+")):
                for part in name.split("+"):
                    signal, group, weight = part.split(":")
                    rows["base"] = rank_group(rows, signal, group, float(weight))
                pred[name] = rows.base
                rows["base"] = scores["base"]
                metrics.append(dict(day=day, name=name, mode="blend", ap=average_precision_score(rows.error, pred[name])))
        predictions.append(pred)
    metrics = pd.DataFrame(metrics)
    blended = metrics[metrics["mode"] == "blend"].pivot(index="name", columns="day", values="ap").dropna()
    blended["mean"] = blended[DAYS].mean(1)
    print(blended.sort_values("mean", ascending=False).round(6).to_string())
    print("Standalone controls")
    print(metrics[metrics["mode"] == "standalone"].pivot(index="name", columns="day", values="ap").round(5))
    metrics.to_csv(OUT / "validation_hypotheses_090734.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(CACHE / "oof.csv", index=False)
    for name in args.candidate:
        if name not in blended.index:
            raise ValueError(f"Unavailable candidate {name}")
        print("CI", name, {day: bootstrap(q, ["base", name])[name].round(5).tolist()
                           for day, q in zip(DAYS, predictions)})
        test = data[data.day == "test"].reset_index(drop=True)
        if any(test[part.split(":")[0]].isna().any() for part in name.split("+")):
            print("No test inference for", name)
            continue
        for part in name.split("+"):
            signal, group, weight = part.split(":")
            test["base"] = rank_group(test, signal, group, float(weight))
        template = pd.read_csv(D / "sample_submission.csv")[["id"]]
        output = template.merge(test[["id", "base"]], on="id", validate="one_to_one").rename(columns={"base": "score"})
        assert len(output) == len(template) == 362 and output.score.between(0, 1).all()
        stem = name.replace(":", "_").replace("+", "_plus_").replace(".", "")
        output.to_csv(OUT / f"submission_090734_{stem}.csv", index=False)
        before = pd.read_csv(BASE_FILE).set_index("id").score.reindex(output.id).to_numpy()
        after = output.score.to_numpy()
        print("test changed", np.count_nonzero(before != after), "top50 overlap",
              len(set(np.argsort(-before)[:50]) & set(np.argsort(-after)[:50])), flush=True)


if __name__ == "__main__":
    main()
