"""Fit a spatial head on frozen DINOv2 tokens to reconstruct manual boxes."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from audit_adapt_dino import candidates, targets, WORK
from audit_uncertainty import OUT, rank_mix
from features import BOX, CLASSES, D, load, iou_matrix


class Refiner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 48), nn.GELU())
        self.head = nn.Sequential(nn.Linear(48*25+len(CLASSES), 128), nn.GELU(),
                                  nn.Dropout(.15), nn.Linear(128, 6))

    def forward(self, tokens, cls):
        patches = self.project(tokens[:, -100:].float()).reshape(-1, 10, 10, 48)
        grid = F.avg_pool2d(patches.permute(0, 3, 1, 2), 2).flatten(1)
        return self.head(torch.cat((grid, cls), dim=1))


def supervision(rows, truth):
    target = np.zeros((len(rows), 4), dtype=np.float32)
    overlap = np.zeros(len(rows), dtype=np.float32)
    boxes = rows[BOX].to_numpy()
    dimensions = np.tile(boxes[:, 2:] - boxes[:, :2], 2)
    known = rows[rows.part != "test"]
    for (frame, cls), group in known.groupby(["frame_id", "class_name"]):
        gt = truth.loc[(truth.frame_id == frame)&(truth.class_name == cls), BOX].to_numpy()
        if not len(gt):
            continue
        idx = group.index.to_numpy()
        matrix = iou_matrix(boxes[idx], gt)
        overlap[idx] = matrix.max(1)
        target[idx] = (gt[matrix.argmax(1)] - boxes[idx])/dimensions[idx]
    return target.clip(-1.5, 1.5), overlap


def fit_predict(tokens, cls, target, overlap, errors, weight, fit, pred, seed):
    torch.manual_seed(seed)
    model = Refiner(tokens.shape[-1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.03)
    for epoch in range(15):
        model.train()
        for idx in fit[torch.randperm(len(fit),device="cuda")].split(64):
            logits = model(tokens[idx], cls[idx])
            valid = overlap[idx] >= .20
            geom = F.smooth_l1_loss(logits[:, :4], target[idx], reduction="none").mean(1)
            obj = F.binary_cross_entropy_with_logits(logits[:,4], (overlap[idx]>.2).float(), reduction="none")
            err = F.binary_cross_entropy_with_logits(logits[:,5], errors[idx], reduction="none")
            loss = (weight[idx]*(geom*valid*4+obj*.4+err)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(tokens[b], cls[b]) for b in pred.split(64)]).cpu().numpy()


def score(rows, predictions, baseline):
    boxes = rows[BOX].to_numpy()
    dimensions = np.tile(boxes[:,2:]-boxes[:,:2],2)
    revised = boxes+predictions[:,:4]*dimensions
    revised = revised.clip([0,0,0,0],[1280,720,1280,720])
    revised[:,2:] = np.maximum(revised[:,2:], revised[:,:2]+1e-6)
    quality = np.diag(iou_matrix(boxes,revised))
    logits = predictions[:,4:].clip(-40,40)
    prob = 1/(1+np.exp(-logits))
    geometry = 1-quality*prob[:,0]
    direct = prob[:,1]
    person_geometry = rank_mix(rows,baseline,geometry,.25,True)
    person_half = rank_mix(rows,baseline,geometry,.50,True)
    odds = np.clip(baseline,.001,.999)/np.clip(1-baseline,.001,.999)
    calibrated = odds**1.25/(1+odds**1.25)
    return {
        "baseline":baseline,
        "geometry_rank":rank_mix(rows,baseline,geometry,.35),
        "direct_rank":rank_mix(rows,baseline,direct,.35),
        "joint_rank":rank_mix(rows,baseline,(geometry+direct)/2,.35),
        "joint_blend":np.where(rows.class_name=="person",baseline,.8*baseline+.2*(geometry+direct)/2),
        "person_geometry":person_geometry,
        "person_geometry_half":person_half,
        "person_geometry_calibrated":np.where(rows.class_name=="person",person_geometry,calibrated),
        "person_direct":rank_mix(rows,baseline,direct,.25,True),
    }


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    train,*_=load("train")
    test,*_=load("test")
    truth=targets()
    synthetic=candidates(truth)
    rows=pd.concat([train.assign(part="train"),test.assign(part="test"),synthetic.assign(part="synth")],ignore_index=True)
    rows=rows[rows.class_name.isin(CLASSES)].reset_index(drop=True)
    cache=torch.load(WORK/"dino_tokens.pt",weights_only=True)
    assert len(cache)==len(rows) and cache.shape[1]>=101
    tokens=cache.cuda()
    del cache
    day=rows.frame_id.str[-15:-7]
    cls=F.one_hot(torch.tensor(rows.class_name.map(CLASSES.index).to_numpy(),device="cuda"),len(CLASSES)).float()
    coords,iou=supervision(rows,truth)
    target,overlap=[torch.as_tensor(x,device="cuda") for x in (coords,iou)]
    error=torch.as_tensor(rows.error.fillna(0).to_numpy(dtype=np.float32),device="cuda")
    weight=torch.as_tensor(np.where(rows.part=="synth",.30,1.).astype(np.float32),device="cuda")
    val=pd.read_csv(WORK/"final_candidates_oof.csv").set_index("id")
    records=[]
    validation=[]
    for held in ["20261025","20261026","test"]:
        if held=="test":
            table=pd.DataFrame(records)
            table.to_csv(OUT/"validation_box_refinement.csv",index=False)
            pd.concat(validation).to_csv(WORK/"box_refinement_oof.csv",index=False)
            table=table.pivot(index="name",columns="day",values="ap")
            table["mean"]=table.mean(axis=1)
            print(table.sort_values("mean",ascending=False).round(6).to_string(),flush=True)
            improved=(table[["20261025","20261026"]]>table.loc["baseline",["20261025","20261026"]]).all(axis=1)
            eligible=table[improved].sort_values("mean",ascending=False)
            if eligible.empty:
                return
            best=eligible.index[0]
            fit_mask=rows.part!="test"
            pred_mask=rows.part=="test"
            base=pd.read_csv(OUT/"submission_multiview_duplicates.csv").set_index("id").score
        else:
            fit_mask=(rows.part!="test") & (day!=held)
            pred_mask=(rows.part=="train") & (day==held)
            base=val.base
        fit,pred=[torch.tensor(np.flatnonzero(m),device="cuda") for m in [fit_mask,pred_mask]]
        predictions=np.mean([fit_predict(tokens,cls,target,overlap,error,weight,fit,pred,s)
                             for s in (117,312,614)],axis=0)
        det=rows.loc[pred_mask].reset_index(drop=True)
        opts=score(det,predictions,det.id.map(base).to_numpy())
        if held!="test":
            validation.append(det[["id", "frame_id", "class_name", "error"]].assign(day=held,**opts))
            for name,values in opts.items():
                ap=average_precision_score(det.error,values)
                records.append(dict(name=name,day=held,ap=ap))
            print(held,[(r["name"],round(r["ap"],5)) for r in records if r["day"]==held],flush=True)
        else:
            template=pd.read_csv(D/"sample_submission.csv")[["id"]]
            out=template.merge(pd.DataFrame(dict(id=det.id,score=opts[best])),on="id",validate="one_to_one")
            assert len(out)==362 and out.score.between(0,1).all()
            path=OUT/f"submission_refinement_{best}.csv"
            out.to_csv(path,index=False)
            print("saved",path,flush=True)


if __name__=="__main__":
    main()
