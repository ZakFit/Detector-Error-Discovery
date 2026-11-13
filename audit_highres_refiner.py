"""Train high resolution spatial and boundary-aware refiners on day-held-out labels."""
import argparse
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import pandas as pd
import torch
import timm
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
import audit_adapt_dino as enc
from audit_box_refinement import supervision
from audit_uncertainty import rank_mix, OUT
from features import BOX, CLASSES, D, load, iou_matrix

WORK = enc.WORK / "highres_refiner"
WORK.mkdir(exist_ok=True)


class Refiner(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.project = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 48), nn.GELU())
        dimension = 48 * (100 if mode == "dense" else 65 if mode == "boundary" else 25)
        self.head = nn.Sequential(nn.Linear(dimension + 4, 128), nn.GELU(),
                                  nn.Dropout(.15), nn.Linear(128, 6))

    def forward(self, tokens, cls):
        grid = self.project(tokens[:, -400:].float()).reshape(-1, 20, 20, 48).permute(0, 3, 1, 2)
        if self.mode == "dense":
            vector = F.avg_pool2d(grid, 2).flatten(1)
        else:
            vector = F.avg_pool2d(grid, 4).flatten(1)
            if self.mode == "boundary":
                vector = torch.cat([vector, grid.mean(2).flatten(1), grid.mean(3).flatten(1)], 1)
        return self.head(torch.cat([vector, cls], 1))


def fit(tokens, cls, target, overlap, errors, weights, train, predict, seed, mode, epochs=15, save=None):
    torch.manual_seed(seed)
    net = Refiner(mode).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0006, weight_decay=.03)
    for epoch in range(epochs):
        net.train()
        for idx in train[torch.randperm(len(train))].split(48):
            logits = net(tokens[idx].cuda(), cls[idx])
            geom = F.smooth_l1_loss(logits[:,:4], target[idx], reduction="none").mean(1)
            obj = F.binary_cross_entropy_with_logits(logits[:,4], (overlap[idx]>.2).float(), reduction="none")
            err = F.binary_cross_entropy_with_logits(logits[:,5], errors[idx], reduction="none")
            loss = (weights[idx]*(4*geom*(overlap[idx]>=.2) + .4*obj + err)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    if save is not None:
        torch.save(net.state_dict(), save)
    with torch.inference_mode():
        result = torch.cat([net.eval()(tokens[b].cuda(), cls[b]) for b in predict.split(48)])
    return result.cpu().numpy()


def geometry(rows, prediction):
    bb = rows[BOX].to_numpy()
    revised = (bb+prediction[:,:4]*np.tile(bb[:,2:]-bb[:,:2], 2)).clip([0,0,0,0],[1280,720,1280,720])
    revised[:,2:] = np.maximum(revised[:,2:], revised[:,:2]+1e-6)
    quality = np.diag(iou_matrix(bb,revised))
    return 1-quality/(1+np.exp(-prediction[:,4].clip(-40,40)))


def variants(det, base, signals):
    scores = {"baseline":base}
    for name, signal in signals.items():
        for weight in (.25,.50):
            scores[f"{name}_rank_{weight}"] = rank_mix(det, base, signal, weight, True)
        scores[f"{name}_direct_025"] = np.where(det.class_name=="person",.75*base+.25*signal,base)
    return scores


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--test", nargs="*")
    args = cli.parse_args()
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("high")
    train,*_=load("train")
    test,*_=load("test")
    truth=enc.targets()
    rows=pd.concat([train.assign(part="train"), test.assign(part="test"),
                    enc.candidates(truth).assign(part="synth")], ignore_index=True)
    rows=rows[rows.class_name.isin(CLASSES)].reset_index(drop=True)
    path=WORK/"tokens280.pt"
    if not path.exists():
        enc.SIZE=280
        enc.BATCH=12
        backbone=timm.create_model("vit_large_patch14_dinov2.lvd142m",pretrained=True,num_classes=0,img_size=280).eval().cuda()
        print("Extracting",len(rows),"high resolution crops",flush=True)
        # Preserve the original crop geometry so resolution is the controlled change.
        chunks=[]
        for start in range(0,len(rows),240):
            chunks.append(enc.extract(backbone,rows.iloc[start:start+240]))
            print("extracted",min(start+240,len(rows)),flush=True)
        torch.save(torch.cat(chunks),path)
        del backbone,chunks
        torch.cuda.empty_cache()
    tokens=torch.load(path,weights_only=True)
    assert tokens.shape==(len(rows),401,1024)
    # Keep the cache in CPU memory. Only minibatches reside on the GPU.
    cls=F.one_hot(torch.tensor(rows.class_name.map(CLASSES.index).to_numpy(),device="cuda"),4).float()
    coords,iou=supervision(rows,truth)
    target,overlap=[torch.as_tensor(x,device="cuda") for x in (coords,iou)]
    errors=torch.tensor(rows.error.fillna(0).to_numpy(),device="cuda",dtype=torch.float32)
    weights=torch.tensor(np.where(rows.part=="synth",.30,1.),device="cuda",dtype=torch.float32)
    day=rows.frame_id.str[-15:-7]
    old=pd.read_csv(enc.WORK/"box_refinement_oof.csv").set_index("id").person_geometry_half
    metrics, outputs=[],[]
    days=["test"] if args.test else ["20261025","20261026"]
    for held in days:
        fitting=(rows.part!="test") & ((day!=held) if held!="test" else True)
        predicting=(rows.part=="test") if held=="test" else ((rows.part=="train")&(day==held))
        fit_idx,pred_idx=[torch.tensor(np.flatnonzero(x),device="cpu") for x in (fitting,predicting)]
        det=rows[predicting].reset_index(drop=True)
        base=det.id.map(pd.read_csv(OUT/"submission_refinement_person_geometry_half.csv").set_index("id").score if held=="test" else old).to_numpy()
        signals={}
        for mode in (args.test or ["coarse","dense","boundary"]):
            predictions=[]
            for seed in (117,312,614):
                seed_file=WORK/f"{held}_{mode}_{seed}.npy"
                if seed_file.exists():
                    prediction=np.load(seed_file)
                else:
                    prediction=fit(tokens,cls,target,overlap,errors,weights,fit_idx,pred_idx,seed,mode,
                                   save=WORK/f"all_{mode}_{seed}.pt" if held=="test" else None)
                    np.save(seed_file,prediction)
                predictions.append(prediction)
                print("trained",held,mode,seed,flush=True)
            signals[mode]=geometry(det,np.mean(predictions,axis=0))
        scores=variants(det,base,signals)
        output=det[["id","frame_id","class_name","error"]].assign(day=held,**signals,**scores)
        outputs.append(output)
        if held!="test":
            for name,score in scores.items():
                metrics.append(dict(day=held,name=name,ap=average_precision_score(det.error,score)))
            print(pd.DataFrame(metrics).pivot(index="name",columns="day",values="ap").round(6).to_string(),flush=True)
            pd.DataFrame(metrics).to_csv(OUT/"validation_highres_refiner.csv",index=False)
        pd.concat(outputs,ignore_index=True).to_csv(WORK/("test.csv" if held=="test" else "oof.csv"),index=False)


if __name__=="__main__":
    main()
