"""Compare direct geometry blending with the rank-preserving Public baseline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "data" / "gpu_runs"
OUT = ROOT.parent.parent / "outputs"
d = pd.read_csv(WORK / "box_refinement_oof.csv").rename(columns={"person_geometry_half":"score"})
raw = pd.read_csv(WORK / "hypotheses_v1" / "refiner.csv")
d = d.merge(raw[["id","refiner_clean"]], on="id", validate="one_to_one")
d["candidate"] = np.where(d.class_name == "person", .75*d.score+.25*d.refiner_clean, d.score)
metrics=[]
for day, g in d.groupby("day"):
    g=g.reset_index(drop=True)
    y=g.error.to_numpy()
    frames=list(g.groupby("frame_id").indices.values())
    rng=np.random.default_rng(715)
    draws=rng.integers(0,len(frames),size=(2000,len(frames)))
    weights=np.stack([np.bincount(np.concatenate([frames[i] for i in draw]),minlength=len(g)) for draw in draws])
    boot={}
    for col in ["score","candidate"]:
        score=g[col].to_numpy()
        order=np.argsort(-score,kind="stable")
        end=np.flatnonzero(np.r_[score[order][1:]!=score[order][:-1],True])
        hits=np.cumsum(weights[:,order]*y[order],axis=1)[:,end]
        total=np.cumsum(weights[:,order],axis=1)[:,end]
        boot[col]=((hits/np.maximum(total,1))*np.diff(hits,axis=1,prepend=0)).sum(1)/np.maximum(hits[:,-1],1)
    baseline=average_precision_score(y,g.score)
    candidate=average_precision_score(y,g.candidate)
    metrics.append(dict(day=int(day),baseline=baseline,candidate=candidate,delta=candidate-baseline,
                        frame_bootstrap_95pct=np.quantile(boot["candidate"]-boot["score"],[.025,.975]).tolist()))
test=pd.read_csv(ROOT/"data/raw/test.csv")
baseline=pd.read_csv(OUT/"submission_refinement_person_geometry_half.csv")
result=baseline.merge(test[["id","class_name"]],on="id",validate="one_to_one").merge(
    raw[["id","refiner_clean"]],on="id",validate="one_to_one")
original=result.score.to_numpy().copy()
person=(result.class_name=="person").to_numpy()
result["score"]=np.where(person,.75*original+.25*result.refiner_clean,original)
template=pd.read_csv(ROOT/"data/raw/sample_submission.csv")
assert result.id.tolist()==template.id.tolist()
assert len(result)==362 and result.id.is_unique and result.score.between(0,1).all()
assert np.array_equal(original[~person],result.score.to_numpy()[~person])
path=OUT/"submission_person_geometry_direct_025.csv"
result[["id","score"]].to_csv(path,index=False)
comparison={"changed":int((np.abs(original-result.score.to_numpy())>1e-12).sum()),
            "top50_overlap":len(set(np.argsort(-original)[:50])&set(np.argsort(-result.score.to_numpy())[:50])),
            "top100_overlap":len(set(np.argsort(-original)[:100])&set(np.argsort(-result.score.to_numpy())[:100]))}
report={"validation":metrics,"test":comparison,"public_score":None,
        "selection_note":"Exploratory selection among three raw blending weights and nine threshold transforms on two repeatedly reused validation days."}
(WORK/"hypotheses_v1/direct_calibration_report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
print(json.dumps(report,indent=2))
print(path)
