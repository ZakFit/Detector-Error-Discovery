"""Global assignment over independently reconstructed scene boxes."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
from audit_uncertainty import OUT, WORK, VIEWS, signals, rank_mix
from features import D, CLASSES, BOX, load, iou_matrix


def scene_matching(det, ref):
    matched=np.zeros(len(det))
    ref=ref.merge(det[["id","frame_id"]],on="id",validate="many_to_one")
    groups={(f,c,v):q for (f,c,v),q in ref[ref.confidence>=.1].groupby(["frame_id","class_name","view"])}
    for (fid,cls),source in det.groupby(["frame_id","class_name"]):
        idx=source.index.to_numpy()
        bb=source[BOX].to_numpy()
        for view in range(len(VIEWS)):
            teacher=groups.get((fid,cls,view))
            if teacher is None:
                continue
            boxes=teacher[BOX].to_numpy()
            order=np.argsort(-teacher.confidence.to_numpy())
            keep=[]
            while len(order):
                current=order[0]
                keep.append(current)
                order=order[1:]
                if len(order):
                    overlap=iou_matrix(boxes[current:current+1],boxes[order])[0]
                    order=order[overlap<.60]
            matrix=iou_matrix(bb,boxes[keep])
            ok=matrix>=.6
            r,c=linear_sum_assignment(np.where(ok,len(idx)+1+matrix,0),maximize=True)
            valid=ok[r,c]
            matched[idx[r[valid]]]+=1/len(VIEWS)
    return 1-matched


def run(det,base,fold):
    ref=pd.read_csv(WORK/f"uncertainty_{fold}.csv")
    s=signals(det,ref)
    assignment=scene_matching(det,ref)
    small=det.class_name.to_numpy()!="person"
    rank=rank_mix(det,base,s["probability"],.25)
    duplicate_scores=np.clip(rank+.12*s["duplicate"]*(1-rank),0,1)
    return {
        "base":base,
        "probability":rank,
        "probability_and_duplicates":duplicate_scores,
        "selected_keep_person":np.where(small,duplicate_scores,base),
        "scene_assignment":np.where(small,.85*base+.15*assignment,base),
        "scene_rank":rank_mix(det,base,assignment,.25),
        "scene_probability":np.where(small,.9*rank+.1*assignment,rank),
        "consensus":rank_mix(det,base,s["consensus"],.25)
    }


def main():
    train,*_=load("train")
    val=pd.read_csv(WORK/"validation_predictions.csv").set_index("id")
    entries=[]
    predictions=[]
    for fold,day in [("d26","20261025"),("d25","20261026")]:
        det=train[train.class_name.isin(CLASSES)&(train.frame_id.str[-15:-7]==day)].reset_index(drop=True)
        base=np.where(det.class_name=="person",det.id.map(val.current_0842),det.id.map(val.crop_reliability))
        variants=run(det,base,fold)
        predictions.append(det[["id","error","class_name","frame_id"]].assign(day=day,**variants))
        for name,score in variants.items():
            entries.append({"name":name,"day":day,"ap":average_precision_score(det.error,score)})
    table=pd.DataFrame(entries)
    pivot=table.pivot(index="name",columns="day",values="ap")
    pivot["mean"]=pivot.mean(axis=1)
    print(pivot.sort_values("mean",ascending=False).round(6).to_string(),flush=True)
    table.to_csv(OUT/"validation_scene_matching.csv",index=False)
    pd.concat(predictions).to_csv(WORK/"scene_oof.csv",index=False)
    best=pivot[(pivot[["20261025","20261026"]]>pivot.loc["base",["20261025","20261026"]]).all(axis=1)].sort_values("mean",ascending=False).index[0]
    test,*_=load("test")
    test=test[test.class_name.isin(CLASSES)].reset_index(drop=True)
    base=pd.read_csv(OUT/"submission_crop_person_0842.csv").set_index("id").score
    options=run(test,test.id.map(base).to_numpy(),"all")
    for name in dict.fromkeys(["probability",best,"selected_keep_person"]):
        template=pd.read_csv(D/"sample_submission.csv")[["id"]]
        out=template.merge(pd.DataFrame({"id":test.id,"score":options[name]}),on="id",validate="one_to_one")
        assert len(out)==362 and out.score.between(0,1).all() and np.isfinite(out.score).all()
        filename="submission_multiview_duplicates.csv" if name=="selected_keep_person" else f"submission_scene_{name}.csv"
        out.to_csv(OUT/filename,index=False)
        print("saved",name,flush=True)


if __name__=="__main__":
    main()
