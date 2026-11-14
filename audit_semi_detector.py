"""Controlled detector self-training on day-disjoint machine crops."""
import argparse
import shutil
import json
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import average_precision_score
from audit_gpu_yolo import YOLO, WORK, window, infer_crops
from audit_gpu_eval import yolo_signal
from audit_uncertainty import rank_mix, OUT
from features import D, BOX, CLASSES, load, iou_matrix


def pseudo_dataset(fold):
    directory=WORK/f"pseudo_{fold}"
    directory.mkdir(exist_ok=True)
    manifest=directory/"accepted.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    images,labels=directory/"images",directory/"labels"
    images.mkdir(exist_ok=True); labels.mkdir(exist_ok=True)
    pool=pd.read_csv(D/"machine_predictions.csv")
    excluded=set()
    for part in ["train","test"]:
        ctx=pd.read_csv(D/f"{part}_context.csv")
        excluded.update(ctx[["frame_id","previous","next"]].stack())
    pool=pool[pool.class_name.isin(["nude_hands","cup","tray"]) & ~pool.frame_id.isin(excluded)]
    if fold!="all":
        pool=pool[pool.frame_id.str[-9:-7]==fold[-2:]]
    # Spread candidates over scene times and classes without using error labels.
    pool=pd.concat([g.sample(min(180,len(g)),random_state=514) for _,g in pool.groupby("class_name")])
    model=YOLO(str(WORK/f"crop_{fold}/weights/last.pt"))
    accepted=[]
    for fid,rows in pool.groupby("frame_id"):
        image=cv2.imread(str(D/f"{fid}.jpg"))
        sx,sy=image.shape[1]/1280,image.shape[0]/720
        windows=[window(b,4) for b in rows[BOX].to_numpy()]
        patches=[image[round(y1*sy):round(y2*sy),round(x1*sx):round(x2*sx)] for x1,y1,x2,y2 in windows]
        a=model.predict(patches,imgsz=384,device=0,conf=.15,verbose=False)
        b=model.predict([np.ascontiguousarray(p[:,::-1]) for p in patches],imgsz=384,device=0,conf=.15,verbose=False)
        for row,patch,pred,rev in zip(rows.itertuples(),patches,a,b):
            first=pred.boxes.data.cpu().numpy()
            second=rev.boxes.data.cpu().numpy()
            if len(first)==0 or len(second)==0:
                continue
            second[:,[0,2]]=patch.shape[1]-second[:,[2,0]]
            overlaps=iou_matrix(first[:,:4],second[:,:4])
            same=first[:,5,None]==second[None,:,5]
            overlap=overlaps*same
            # Reject crops containing a moderate-confidence object unconfirmed by the other view.
            if (overlap.max(1)<.65).any() or (overlap.max(0)<.65).any():
                continue
            matched=second[overlap.argmax(1)]
            confidence=np.minimum(first[:,4],matched[:,4])
            expected=CLASSES.index(row.class_name)
            if not ((first[:,5]==expected)&(confidence>=.55)).any():
                continue
            if (confidence<.25).any():
                continue
            boxes=(first[:,:4]+matched[:,:4])/2
            center=(boxes[:,:2]+boxes[:,2:])/2/[patch.shape[1],patch.shape[0]]
            size=(boxes[:,2:]-boxes[:,:2])/[patch.shape[1],patch.shape[0]]
            path=images/f"{row.detection_id}.jpg"
            cv2.imwrite(str(path),patch)
            lines=[f"{int(cls)} "+" ".join(f"{v:.7f}" for v in values)
                   for cls,values in zip(first[:,5],np.column_stack([center,size]))]
            (labels/f"{path.stem}.txt").write_text("\n".join(lines))
            accepted.append(str(path.resolve()))
    manifest.write_text(json.dumps(accepted,indent=2))
    print(fold,"pseudo crops accepted",len(accepted),"of",len(pool),flush=True)
    return accepted


def train(fold,arm,pseudo):
    cfgdir=WORK/f"semisupervised_{fold}_{arm}"
    cfgdir.mkdir(exist_ok=True)
    human=sorted((WORK/f"dataset_crop_{fold}/images").glob("*.jpg"))
    human=[str(p.resolve()) for p in human]
    rng=np.random.default_rng(511)
    extra=pseudo if arm=="pseudo" else list(rng.choice(human,len(pseudo)))
    selected=human*3+extra
    listing=cfgdir/"train.txt"
    listing.write_text("\n".join(selected),encoding="utf-8")
    cfg=cfgdir/"dataset.yaml"
    cfg.write_text(f"path: {cfgdir.as_posix()}\ntrain: {listing.as_posix()}\nval: {listing.as_posix()}\nnames: {dict(enumerate(CLASSES))}\n")
    ckpt=WORK/f"semi_{fold}_{arm}/weights/last.pt"
    if not ckpt.exists():
        model=YOLO(str(WORK/f"crop_{fold}/weights/last.pt"))
        model.train(data=str(cfg),epochs=10,imgsz=384,batch=12,workers=0,device=0,
                    amp="bf16",optimizer="AdamW",lr0=.00015,lrf=.2,warmup_epochs=0,
                    freeze=10,seed=515,patience=0,val=False,plots=False,verbose=False,
                    mosaic=0,scale=.25,fliplr=.5,project=str(WORK),name=f"semi_{fold}_{arm}",
                    exist_ok=True,save=True)
    return YOLO(str(ckpt))


def evaluate():
    tr,*_=load("train")
    tr=tr[tr.class_name.isin(CLASSES)]
    previous=pd.read_csv(WORK/"scene_oof.csv").set_index("id")
    before_crop=pd.read_csv(D.parent/"audit_oof.csv").set_index("id").ft_reliability_person
    records=[]; oof=[]
    for fold,day in [("d26","20261025"),("d25","20261026")]:
        det=tr[tr.frame_id.str[-15:-7]==day].reset_index(drop=True)
        base=det.id.map(previous.base).to_numpy()
        latest=np.where(det.class_name=="person",base,det.id.map(previous.probability_and_duplicates))
        opts={"base_086558":base,"multiview":latest}
        for arm in ["control","pseudo"]:
            quality,conf=yolo_signal(det,WORK/f"semi_{fold}_{arm}.csv",crop=True)
            weight=.12+.25*conf
            small=det.class_name.to_numpy()!="person"
            opts[f"replace_{arm}"]=np.where(small,(1-weight)*det.id.map(before_crop).to_numpy()+weight*(1-quality),base)
            for prefix,ref in [("base",base),("multiview",latest)]:
                opts[f"{prefix}_{arm}"]=rank_mix(det,ref,1-quality,.25)
        for name,score in opts.items():
            records.append({"name":name,"day":day,"ap":average_precision_score(det.error,score)})
        oof.append(det[["id","frame_id","class_name","error"]].assign(day=day,**opts))
    pd.concat(oof).to_csv(WORK/"semi_oof.csv",index=False)
    result=pd.DataFrame(records)
    result.to_csv(OUT/"validation_semisupervised.csv",index=False)
    pivot=result.pivot(index="name",columns="day",values="ap")
    pivot["mean"]=pivot.mean(axis=1)
    print(pivot.sort_values("mean",ascending=False).round(6).to_string(),flush=True)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("folds",nargs="*",default=["d26","d25"])
    args=cli.parse_args()
    tr,*_=load("train"); te,*_=load("test")
    for fold in args.folds:
        pseudo=pseudo_dataset(fold)
        det=te if fold=="all" else tr[tr.frame_id.str[-9:-7]!=fold[-2:]]
        det=det[det.class_name.isin(CLASSES)]
        for arm in ["control","pseudo"]:
            output=WORK/f"semi_{fold}_{arm}.csv"
            if output.exists():
                continue
            model=train(fold,arm,pseudo)
            infer_crops(model,det,output)
            print("saved",output,flush=True)
    if "all" not in args.folds:
        evaluate()


if __name__=="__main__":
    main()
