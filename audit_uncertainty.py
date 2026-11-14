"""Estimate localization uncertainty from independent transformed views."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np
import pandas as pd
import cv2
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from audit_gpu_yolo import YOLO, WORK, window
from features import BOX, CLASSES, D, load

OUT = D.parent.parent.parent.parent / "outputs"
VIEWS = [(3,384,False),(3,512,True),(4,384,False),(4,512,True),(5,512,False),(5,384,True)]


def predict(det, fold):
    output = WORK / f"uncertainty_{fold}.csv"
    if output.exists():
        return pd.read_csv(output)
    crop_model = YOLO(str(WORK / f"crop_{fold}/weights/last.pt"))
    full_model = YOLO(str(WORK / f"full_{fold}/weights/last.pt"))
    records = []
    for fid, group in det.groupby("frame_id"):
        image = cv2.imread(str(D / f"{fid}.jpg"))
        for view, (ratio, size, flip) in enumerate(VIEWS):
            small = group[group.class_name != "person"]
            if len(small):
                windows = [window(b, ratio) for b in small[BOX].to_numpy()]
                patches = [image[y1:y2,x1:x2] for x1,y1,x2,y2 in windows]
                pred = crop_model.predict([np.ascontiguousarray(p[:,::-1]) if flip else p for p in patches],
                                          imgsz=size,device=0,verbose=False,conf=0.03)
                for ident, bounds, res in zip(small.id, windows, pred):
                    x,y,xmax,_ = bounds
                    for x1,y1,x2,y2,c,k in res.boxes.data.cpu().tolist():
                        if flip:
                            x1,x2 = xmax-x-x2, xmax-x-x1
                        records.append((ident,view,CLASSES[int(k)],c,x+x1,y+y1,x+x2,y+y2))
            people = group[group.class_name == "person"]
            if len(people):
                res = full_model.predict(np.ascontiguousarray(image[:,::-1]) if flip else image,
                                         imgsz=[768,960,1280][view//2], device=0,verbose=False,conf=0.03)[0]
                for x1,y1,x2,y2,c,k in res.boxes.data.cpu().tolist():
                    if int(k) != CLASSES.index("person"):
                        continue
                    if flip:
                        x1,x2 = 1280-x2,1280-x1
                    for ident in people.id:
                        records.append((ident,view,"person",c,x1,y1,x2,y2))
        print(fold,fid,flush=True)
    frame = pd.DataFrame(records, columns=["id","view","class_name","confidence",*BOX])
    frame.to_csv(output,index=False)
    return frame


def signals(det, ref):
    from features import iou_matrix
    groups = {(i,v):g for (i,v),g in ref.groupby(["id","view"])}
    mean_iou, success, weighted, certainty, boxes, consensus = [], [], [], [], [], []
    for row in det.itertuples():
        bb = np.array([[getattr(row,b) for b in BOX]])
        overlaps, probs, scores, matches = [], [], [], []
        for view in range(len(VIEWS)):
            g = groups.get((row.id,view))
            g = g[g.class_name == row.class_name] if g is not None else None
            if g is None or g.empty:
                overlaps.append(0); probs.append(0); scores.append(0); continue
            ious = iou_matrix(bb,g[BOX].to_numpy())[0]
            # Prefer spatial agreement, allow confidence to break competing matches.
            j = np.argmax(ious * np.sqrt(g.confidence.to_numpy()))
            iou, conf = ious[j], g.confidence.iloc[j]
            overlaps.append(iou)
            probs.append(expit((iou-0.6)/0.08))
            scores.append(conf)
            if iou > 0.25 and conf > 0.1:
                matches.append(g[BOX].to_numpy()[j])
        mean_iou.append(1-np.mean(overlaps))
        success.append(1-np.mean(probs))
        weighted.append(1-np.average(probs,weights=np.array(scores)+0.05))
        certainty.append(np.std(overlaps))
        boxes.append(np.median(matches,axis=0) if matches else bb[0])
        consensus.append(1-float(iou_matrix(bb,np.array(boxes[-1])[None])[0,0]) if matches else 1.)
    # Soft penalty for several source boxes competing for one common reconstructed object.
    duplicate = np.zeros(len(det))
    boxes = np.stack(boxes)
    for _,group in det.groupby(["frame_id","class_name"]):
        ii = group.index.to_numpy()
        if len(ii)<2:
            continue
        overlaps = iou_matrix(boxes[ii],boxes[ii])
        for a in range(len(ii)):
            better = np.asarray(success)[ii] < success[ii[a]]
            if better.any() and overlaps[a,better].max() > 0.75:
                duplicate[ii[a]] = overlaps[a,better].max()
    return {"mean":np.array(mean_iou),"probability":np.array(success),
            "weighted_probability":np.array(weighted),"instability":np.array(certainty),
            "consensus":np.array(consensus),"duplicate":duplicate}


def rank_mix(det, base, evidence, weight, people=False):
    result = base.copy()
    for cls,group in det.groupby("class_name"):
        if (cls == "person") != people:
            continue
        idx = group.index.to_numpy()
        r = (1-weight)*rankdata(base[idx])+weight*rankdata(evidence[idx])
        order = np.argsort(r,kind="stable")
        assigned=np.empty(len(idx))
        assigned[order]=np.sort(base[idx])
        result[idx]=assigned
    return result


def options(det, base, sig):
    results={"baseline_086558":base}
    small=det.class_name.to_numpy()!="person"
    for signal in ["mean","probability","weighted_probability","consensus"]:
        for weight in [0.25,0.5]:
            results[f"rank_{signal}_{weight}"]=rank_mix(det,base,sig[signal],weight)
        results[f"blend_{signal}"]=np.where(small,.8*base+.2*sig[signal],base)
    results["rank_person_probability"]=rank_mix(det,base,sig["probability"],.25,True)
    results["duplicates"]=np.clip(base+.12*sig["duplicate"]*(1-base),0,1)
    return results


def main():
    train,*_=load("train")
    test,*_=load("test")
    train=train[train.class_name.isin(CLASSES)].reset_index(drop=True)
    test=test[test.class_name.isin(CLASSES)].reset_index(drop=True)
    val=pd.read_csv(WORK/"validation_predictions.csv").set_index("id")
    metrics, oofs=[],[]
    for fold,day in [("d26","20261025"),("d25","20261026")]:
        det=train[train.frame_id.str[-15:-7]==day].reset_index(drop=True)
        base=np.where(det.class_name=="person",det.id.map(val.current_0842),det.id.map(val.crop_reliability))
        sig=signals(det,predict(det,fold))
        scores=options(det,base,sig)
        oofs.append(det[["id","frame_id","class_name","error"]].assign(day=day,**scores))
        for name,score in scores.items():
            metrics.append({"name":name,"day":day,"ap":average_precision_score(det.error,score)})
    metrics=pd.DataFrame(metrics)
    table=metrics.pivot(index="name",columns="day",values="ap")
    table["mean"]=table.mean(axis=1)
    print(table.sort_values("mean",ascending=False).round(5).to_string(),flush=True)
    metrics.to_csv(OUT/"validation_uncertainty.csv",index=False)
    pd.concat(oofs).to_csv(WORK/"uncertainty_oof.csv",index=False)
    improved=(table[["20261025","20261026"]]>table.loc["baseline_086558",["20261025","20261026"]]).all(axis=1)
    eligible=table[improved].sort_values("mean",ascending=False)
    if eligible.empty:
        print("No improvement on both days",flush=True)
        return
    best=eligible.index[0]
    ref=predict(test,"all")
    base=pd.read_csv(OUT/"submission_crop_person_0842.csv").set_index("id").score
    result=options(test,test.id.map(base).to_numpy(),signals(test,ref))[best]
    template=pd.read_csv(D/"sample_submission.csv")[["id"]]
    output=template.merge(pd.DataFrame({"id":test.id,"score":result}),on="id",validate="one_to_one")
    assert len(output)==362 and output.score.between(0,1).all() and np.isfinite(output.score).all()
    path=OUT/f"submission_uncertainty_{best}.csv"
    output.to_csv(path,index=False)
    print("Selected",best,"saved",path,flush=True)


if __name__=="__main__":
    main()
