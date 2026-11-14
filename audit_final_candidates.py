import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from audit_scene_matching import scene_matching
from audit_uncertainty import signals,rank_mix,OUT,WORK
from features import load,CLASSES
from scipy.special import logit,expit

def candidates(det,base,ref):
 s=signals(det,ref)
 assignment=scene_matching(det,ref)
 small=det.class_name.to_numpy()!="person"
 out={"base":base}
 out["global_assignment"]=np.where(small,.9*base+.1*assignment,base)
 out["global_rank"]=rank_mix(det,base,assignment,.25)
 out["box_consensus"]=rank_mix(det,base,s["consensus"],.25)
 # Confidence calibration sharpens the small-object score around the evaluation threshold.
 out["threshold_sharpen"]=np.where(small,expit(1.25*logit(np.clip(base,.001,.999))),base)
 out["threshold_soften"]=np.where(small,expit(.8*logit(np.clip(base,.001,.999))),base)
 out["mean_consensus"]=rank_mix(det,base,.5*(s["consensus"]+s["mean"]),.25)
 out["joint_assignment"]=np.where(small,.9*out["threshold_sharpen"]+.1*assignment,base)
 return out

tr,*_=load("train")
val=pd.read_csv(WORK/"scene_oof.csv").set_index("id")
results=[]; preds=[]
for fold,day in [("d26","20261025"),("d25","20261026")]:
 d=tr[tr.class_name.isin(CLASSES)&(tr.frame_id.str[-15:-7]==day)].reset_index(drop=True)
 base=np.where(d.class_name=="person",d.id.map(val.base),d.id.map(val.probability_and_duplicates))
 opts=candidates(d,base,pd.read_csv(WORK/f"uncertainty_{fold}.csv"))
 for n,v in opts.items(): results.append({"name":n,"day":day,"ap":average_precision_score(d.error,v)})
 preds.append(d[["id","frame_id","error"]].assign(day=day,**opts))
r=pd.DataFrame(results); r.to_csv(OUT/"validation_final_candidates.csv",index=False)
p=r.pivot(index="name",columns="day",values="ap")
p["mean"]=p.mean(axis=1)
print(p.sort_values("mean",ascending=False).to_string(),flush=True)
pd.concat(preds).to_csv(WORK/"final_candidates_oof.csv",index=False)
eligible=p[(p[["20261025","20261026"]]>p.loc["base",["20261025","20261026"]]).all(axis=1)].sort_values("mean",ascending=False)
te,*_=load("test");te=te[te.class_name.isin(CLASSES)].reset_index(drop=True)
base=pd.read_csv(OUT/"submission_multiview_duplicates.csv").set_index("id").score
opts=candidates(te,te.id.map(base).to_numpy(),pd.read_csv(WORK/"uncertainty_all.csv"))
for n in eligible.index[:2]:
 template=pd.read_csv("data/raw/sample_submission.csv")[["id"]]
 sub=template.merge(pd.DataFrame({"id":te.id,"score":opts[n]}),on="id",validate="one_to_one")
 assert len(sub)==362 and sub.score.between(0,1).all() and np.isfinite(sub.score).all()
 sub.to_csv(OUT/f"submission_final_{n}.csv",index=False)
 print("saved",n,flush=True)
