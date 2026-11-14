"""Semantic verification, amodal regression and pairwise ranking on fixed image contexts."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from audit_hypotheses_features import CACHE
from features import BOX, CLASSES, iou_matrix

CONTEXT = list("abcd")
CLASSES_ONEHOT = {c: i for i, c in enumerate(CLASSES)}


def train_masks(rows, day):
    eligible = rows.part != "test"
    if day != "test":
        eligible &= rows.frame_id.str[-15:-7] != day
    predict = rows.part == ("test" if day == "test" else "train")
    if day != "test":
        predict &= rows.frame_id.str[-15:-7] == day
    return eligible.to_numpy(), predict.to_numpy()


def semantic(rows, embeddings, gt, train, predict):
    # Class negatives include explicitly annotated confusing objects and detector false positives.
    cat = rows.loc[train & (rows.part == "gt") & (rows.view == 0), "class_name"].value_counts().to_dict()
    label = rows.class_name.where(rows.class_name.map(cat).fillna(0) >= 5, "other").to_numpy().copy()
    real = np.flatnonzero(train & (rows.part == "train"))
    for idx in real:
        row = rows.iloc[idx]
        known = gt[(gt.frame_id == row.frame_id) & (gt.class_name != "table")]
        overlap = iou_matrix(rows.loc[[idx], BOX].to_numpy(), known[BOX].to_numpy())[0]
        same = (known.class_name == row.class_name).to_numpy()
        if overlap[same].max(initial=0) >= .28:
            label[idx] = row.class_name
        elif overlap.max(initial=0) >= .50:
            name = known.class_name.iloc[overlap.argmax()]
            label[idx] = name if cat.get(name, 0) >= 5 else "other"
        else:
            label[idx] = "background"
    train = train & (~rows.occluded) & (rows.view == 0)
    x = normalize(embeddings.astype(np.float32))
    idx = np.flatnonzero(train)
    actual = label[idx]
    fit = LogisticRegression(C=18, class_weight="balanced", max_iter=500)
    fit.fit(x[idx], actual, sample_weight=np.where(rows.part.iloc[idx] == "train", 1.0, .65))
    logits = fit.decision_function(x[predict])
    columns = {name: j for j, name in enumerate(fit.classes_)}
    desired = np.array([columns[c] for c in rows.loc[predict, "class_name"]])
    logit = logits[np.arange(len(logits)), desired].copy()
    logits[np.arange(len(logits)), desired] = -np.inf
    return logits.max(axis=1) - logit


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 48), nn.GELU())

    def forward(self, x):
        patches = self.project(x[:, -100:].float()).reshape(-1, 10, 10, 48)
        return F.avg_pool2d(patches.permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)


class AmodalRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialEncoder()
        self.head = nn.Sequential(nn.Linear(48 * 25 + 2, 128), nn.GELU(), nn.Dropout(.12), nn.Linear(128, 4))

    def forward(self, x, scale):
        return self.head(torch.cat([self.spatial(x).flatten(1), scale], 1))


def rescale(boxes, contexts):
    return (boxes - np.tile(contexts[:, :2], 2)) / np.tile(contexts[:, 2:] - contexts[:, :2], 2)


def array(x):
    return torch.as_tensor(np.asarray(x), device="cuda", dtype=torch.float32)


def amodal(rows, tokens, gt, train, predict, seed, augmented):
    torch.manual_seed(seed)
    fit = train & (rows.class_name == "person") & (rows.part == "gt")
    clean = np.flatnonzero(fit & ~rows.occluded)
    covered = rows[fit & rows.occluded].reset_index().set_index(["frame_id", "view", *BOX])
    clean_keys = pd.MultiIndex.from_frame(rows.loc[clean, ["frame_id", "view", *BOX]])
    mate = covered["index"].reindex(clean_keys).fillna(-1).to_numpy(dtype=int)
    mate = torch.as_tensor(mate, device="cuda")
    ctx = rows[CONTEXT].to_numpy()
    scale = array(np.log((ctx[:, 2:] - ctx[:, :2]) / [1280, 720]))
    labels = array(rescale(rows[BOX].to_numpy(), ctx))
    model = AmodalRegressor().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.03)
    source = torch.as_tensor(clean, device="cuda")
    for _ in range(24):
        for ix in torch.randperm(len(source), device="cuda").split(32):
            idx = source[ix]
            if augmented:
                choice = (mate[ix] >= 0) & (torch.rand(len(ix), device="cuda") < .50)
                idx = torch.where(choice, mate[ix], idx)
            loss = F.smooth_l1_loss(model(tokens[idx], scale[idx]), labels[idx], beta=.12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    idx = np.flatnonzero(predict)
    with torch.inference_mode():
        relative = model.eval()(tokens[idx], scale[idx]).cpu().numpy()
    local = ctx[idx]
    boxes = (relative * np.tile(local[:, 2:] - local[:, :2], 2) + np.tile(local[:, :2], 2))
    boxes = boxes.clip([0, 0, 0, 0], [1280, 720, 1280, 720])
    boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2] + 1e-6)
    return 1 - np.diag(iou_matrix(rows.loc[predict, BOX].to_numpy(), boxes))


class BoxCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialEncoder()
        self.head = nn.Sequential(nn.Linear(48 * 3 + 4 + len(CLASSES), 128), nn.GELU(),
                                  nn.Dropout(.10), nn.Linear(128, 1))
        self.register_buffer("edges", torch.linspace(0, 1, 6))

    def forward(self, tokens, boxes, cls):
        grid = self.spatial(tokens)
        widths = (torch.minimum(boxes[..., 2:3], self.edges[1:]) -
                  torch.maximum(boxes[..., 0:1], self.edges[:-1])).clamp(0, .2)
        heights = (torch.minimum(boxes[..., 3:4], self.edges[1:]) -
                   torch.maximum(boxes[..., 1:2], self.edges[:-1])).clamp(0, .2)
        area = (heights[..., :, None] * widths[..., None, :]).flatten(2) * 25
        inside = (area @ grid) / area.sum(2, keepdim=True).clamp(min=.1)
        outside = ((1 - area) @ grid) / (1 - area).sum(2, keepdim=True).clamp(min=.1)
        context = grid.mean(1)[:, None].expand_as(inside)
        return self.head(torch.cat([inside, outside, context, boxes, cls[:, None].expand(-1, boxes.shape[1], -1)],
                                   dim=2)).squeeze(2)


def critic_data(rows, gt):
    rng = np.random.default_rng(722)
    wh = np.maximum(rows[BOX[2:]].to_numpy() - rows[BOX[:2]].to_numpy(), 4)
    box = rows[BOX].to_numpy()
    middle = (box[:, :2] + box[:, 2:]) / 2
    proposals = [box]
    for sigma in (.11, .15, .22, .28, .34, .43, .53):
        shift = middle + rng.normal(0, sigma, wh.shape) * wh
        half = wh / 2 * np.exp(rng.normal(0, sigma, wh.shape))
        proposals.append(np.column_stack([shift-half, shift+half]).clip([0,0,0,0], [1280,720,1280,720]))
    proposals = np.stack(proposals, axis=1)
    quality = np.zeros(proposals.shape[:2])
    truth = {k: g[BOX].to_numpy() for k, g in gt.groupby(["frame_id", "class_name"])}
    for i, row in enumerate(rows.itertuples()):
        same = truth.get((row.frame_id, row.class_name))
        if same is not None:
            quality[i] = iou_matrix(proposals[i], same).max(1)
    label = (quality < .6).astype(float)
    real = rows.part == "train"
    label[real, 0] = rows.loc[real, "error"]
    relative = np.stack([rescale(x, rows[CONTEXT].to_numpy()) for x in proposals.transpose(1,0,2)], 1)
    classes = F.one_hot(torch.tensor(rows.class_name.map(CLASSES_ONEHOT).fillna(0).to_numpy(dtype=np.int64),
                                      device="cuda"), len(CLASSES)).float()
    return array(relative), array(quality), array(label), classes


def critic(rows, tokens, data, fit, predict, seed, pairwise):
    torch.manual_seed(seed)
    box, quality, label, cls = data
    source = np.flatnonzero(fit & ~rows.occluded & (rows.view == 0) & rows.class_name.isin(CLASSES))
    idx = torch.as_tensor(source, device="cuda")
    real = torch.as_tensor((rows.part == "train").to_numpy(), device="cuda")
    model = BoxCritic().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0007, weight_decay=.025)
    for _ in range(27):
        for batch in idx[torch.randperm(len(idx), device="cuda")].split(40):
            logits = model(tokens[batch], box[batch], cls[batch])
            losses = F.binary_cross_entropy_with_logits(logits, label[batch], reduction="none")
            loss = (losses.mean(1) * torch.where(real[batch], 1.3, .8)).mean()
            if pairwise:
                q = quality[batch]
                err = q < .6
                pairs = err[:, :, None] & (~err[:, None, :])
                differences = logits[:, :, None] - logits[:, None, :]
                # Emphasize near-threshold pairs without discarding larger errors.
                pair_weights = torch.exp(-2.5*((q[:, :, None]-.6).abs() + (q[:, None, :]-.6).abs()))
                loss += (F.softplus(-differences) * pairs * pair_weights).sum() / pairs.sum().clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    indices = torch.tensor(np.flatnonzero(predict), device="cuda")
    with torch.inference_mode():
        return torch.cat([model.eval()(tokens[i], box[i,:1], cls[i])[:, 0] for i in indices.split(64)]).cpu().numpy()


def main():
    from audit_hypotheses_features import metadata
    from audit_hypotheses_scene import DAYS
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("high")
    rows = pd.read_csv(CACHE / "rows.csv")
    _, gt, _ = metadata()
    tokens = torch.load(CACHE / "context.pt", weights_only=True).cuda()
    tight = torch.load(CACHE / "tight.pt", weights_only=True).numpy()
    data = critic_data(rows, gt)
    result = []
    for day in [*DAYS, "test"]:
        train, predict = train_masks(rows, day)
        seed_list = (117, 312, 614)
        scores = {"semantic": semantic(rows, tight, gt, train, predict)}
        print(day, "semantic", flush=True)
        for augmented in (False, True):
            tag = "amodal_aug" if augmented else "amodal_control"
            scores[tag] = np.mean([amodal(rows, tokens, gt, train, predict, seed, augmented) for seed in seed_list], 0)
            print(day, tag, flush=True)
        for pair in (False, True):
            tag = "critic_pair" if pair else "critic_control"
            scores[tag] = np.mean([critic(rows, tokens, data, train, predict, seed, pair) for seed in seed_list], 0)
            print(day, tag, flush=True)
        result.append(rows.loc[predict, ["id"]].assign(**scores))
        pd.concat(result, ignore_index=True).to_csv(CACHE / "visual.csv", index=False)


if __name__ == "__main__":
    main()
