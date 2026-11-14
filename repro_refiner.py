"""Проверка: смешивать ли порядок людей 0.911 (рефайнер Emil) с пайплайном Ruslan.

OOF рефайнера Emil есть только у него локально, поэтому здесь его цепочка для людей повторена на CPU
(упрощение: обучение только на классе person; у Emil — все классы вместе):
  база (люди как в 0.842/0.866) → геометрия DINOv2-L 140 px, rank-mix 0.5 (сабмит 0.907)
                                → рефайнер DINOv2-L 280 px «coarse», rank-mix 0.5 (сабмит 0.911)
Затем к порядку 0.911 подмешивается Ruslan с весом W и считается AP на отложенном дне
(модели учатся на другом дне + синтетических сдвигах ручных боксов, как у Emil).

  python repro_refiner.py      → таблица AP, data/repro_refiner.csv
Кэш признаков DINOv2 — data/dino_person_{140,280}.pt (возобновляемо).
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from compare_ideas import FOLD, ft_file, reorder
from features import BOX, D, iou_matrix, load
from final_ft import ft_agree
from reannotate import agreement, amodal_iou, score

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
torch.set_num_threads(10)


def truth():
    import json
    h = pd.read_csv(D / 'human_boxes_train.csv')
    h['class_name'] = h.class_name.replace(json.load(open(D / 'scoring_policy.json'))['aliases'])
    h = h[h.class_name == 'person'].copy()
    h['x1'], h['x2'] = (h.cx - h.w / 2) * 1280, (h.cx + h.w / 2) * 1280
    h['y1'], h['y2'] = (h.cy - h.h / 2) * 720, (h.cy + h.h / 2) * 720
    return h


def synth(gt):
    """Как audit_adapt_dino.candidates: 12 случайных сдвигов каждого ручного бокса, метка — IoU < 0.6."""
    rng = np.random.default_rng(213)
    rec = []
    for f, g in gt.groupby('frame_id'):
        boxes = g[BOX].to_numpy()
        for box in boxes:
            wh, c0 = box[2:] - box[:2], (box[:2] + box[2:]) / 2
            prop = [box]
            for _ in range(12):
                s = rng.choice([0.10, 0.23, 0.42])
                c, size = c0 + rng.normal(0, s, 2) * wh, wh * np.exp(rng.normal(0, s, 2))
                prop.append(np.r_[c - size / 2, c + size / 2])
            prop = np.clip(prop, 0, [1280, 720, 1280, 720])
            q = iou_matrix(prop, boxes).max(1)
            rec += [(f, 'person', *p, float(v < 0.6)) for p, v in zip(prop, q) if min(p[2:] - p[:2]) >= 4]
    return pd.DataFrame(rec, columns=['frame_id', 'class_name', *BOX, 'error'])


@torch.no_grad()
def tokens(rows, size):
    path = DATA / f'dino_person_{size}.pt'
    if path.exists():
        t = torch.load(path, weights_only=True)
        if len(t) == len(rows):
            return t
    model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=size).eval()
    mean, std = torch.tensor([.485, .456, .406])[None, :, None, None], torch.tensor([.229, .224, .225])[None, :, None, None]
    out, cache, batch = [], {}, []
    n_patch = (size // 14) ** 2
    for i, r in enumerate(rows.itertuples(), 1):
        img = cache.setdefault(r.frame_id, cv2.imread(str(D / f'{r.frame_id}.jpg')))
        dx, dy = (r.x2 - r.x1) * .25, (r.y2 - r.y1) * .25
        x1, y1 = max(0, int(r.x1 - dx)), max(0, int(r.y1 - dy))
        x2, y2 = min(1280, int(np.ceil(r.x2 + dx))), min(720, int(np.ceil(r.y2 + dy)))
        batch.append(cv2.cvtColor(cv2.resize(img[y1:y2, x1:x2], (size, size)), cv2.COLOR_BGR2RGB))
        if len(batch) == 16 or i == len(rows):
            x = (torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2) / 255 - mean) / std
            x = model.blocks[:-2](model.norm_pre(model._pos_embed(model.patch_embed(x))))
            out.append(x[:, -n_patch:].half())
            batch.clear()
            if i % 160 < 16:
                print(size, 'px:', i, '/', len(rows), flush=True)
    t = torch.cat(out)
    torch.save(t, path)
    return t


class Refiner(nn.Module):
    def __init__(self, side):
        super().__init__()
        self.side = side
        self.project = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 48), nn.GELU())
        self.head = nn.Sequential(nn.Linear(48 * 25, 128), nn.GELU(), nn.Dropout(.15), nn.Linear(128, 6))

    def forward(self, t):
        g = self.project(t.float()).reshape(-1, self.side, self.side, 48).permute(0, 3, 1, 2)
        return self.head(F.avg_pool2d(g, self.side // 5).flatten(1))


def fit_predict(t, side, target, overlap, err, w, fit, pred, bs):
    preds = []
    for seed in (117, 312, 614):
        torch.manual_seed(seed)
        net = Refiner(side)
        opt = torch.optim.AdamW(net.parameters(), lr=6e-4, weight_decay=.03)
        for _ in range(15):
            net.train()
            for i in fit[torch.randperm(len(fit))].split(bs):
                lo = net(t[i])
                geom = F.smooth_l1_loss(lo[:, :4], target[i], reduction='none').mean(1)
                obj = F.binary_cross_entropy_with_logits(lo[:, 4], (overlap[i] > .2).float(), reduction='none')
                e = F.binary_cross_entropy_with_logits(lo[:, 5], err[i], reduction='none')
                loss = (w[i] * (4 * geom * (overlap[i] >= .2) + .4 * obj + e)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            preds.append(net.eval()(t[pred]).numpy())
    return np.mean(preds, 0)


def geometry(rows, p):
    bb = rows[BOX].to_numpy()
    rev = (bb + p[:, :4] * np.tile(bb[:, 2:] - bb[:, :2], 2)).clip(0, [1280, 720, 1280, 720])
    rev[:, 2:] = np.maximum(rev[:, 2:], rev[:, :2] + 1e-6)
    return 1 - np.diag(iou_matrix(bb, rev)) / (1 + np.exp(-p[:, 4].clip(-40, 40)))


def main():
    tr, *_ = load('train')
    tr = tr[tr.class_name != 'table'].reset_index(drop=True)
    tr['day'] = tr.frame_id.str[-15:-7]
    tr['agree'], tr['amodal'] = agreement(tr), amodal_iou(tr)
    gt = truth()
    rows = pd.concat([tr[tr.class_name == 'person'].assign(part='train'), synth(gt).assign(part='synth')],
                     ignore_index=True)
    rows['day'] = rows.frame_id.str[-15:-7]
    target, overlap = np.zeros((len(rows), 4), np.float32), np.zeros(len(rows), np.float32)
    bb = rows[BOX].to_numpy().astype(float)
    for f, g in rows.groupby('frame_id'):
        h = gt[gt.frame_id == f][BOX].to_numpy()
        if len(h):
            m = iou_matrix(bb[g.index], h)
            overlap[g.index] = m.max(1)
            target[g.index] = ((h[m.argmax(1)] - bb[g.index]) / np.tile(bb[g.index, 2:] - bb[g.index, :2], 2)).clip(-1.5, 1.5)
    target, overlap = torch.tensor(target), torch.tensor(overlap)
    err = torch.tensor(rows.error.fillna(0).to_numpy(np.float32))
    w = torch.tensor(np.where(rows.part == 'synth', .3, 1.), dtype=torch.float32)
    t140, t280 = tokens(rows, 140), tokens(rows, 280)
    oof = pd.read_csv(DATA / 'ruslan_oof.csv', dtype={'id': str}).set_index('id')
    res, keep = {}, []
    for day, fold in FOLD.items():
        q = tr[tr.day == day].reset_index(drop=True)
        cls, P = q.class_name.values, (q.class_name == 'person').values
        base = (score(q) + 1 - ft_agree(q, ft_file(fold))) / 2
        fit = torch.tensor(np.flatnonzero((rows.day != day).values))
        pi = np.flatnonzero(((rows.part == 'train') & (rows.day == day)).values)
        det = rows.iloc[pi].reset_index(drop=True)
        assert (det.id.values == q.id.values[P]).all()
        g1 = np.zeros(len(q)); g1[P] = geometry(det, fit_predict(t140, 10, target, overlap, err, w, fit, torch.tensor(pi), 64))
        g2 = np.zeros(len(q)); g2[P] = geometry(det, fit_predict(t280, 20, target, overlap, err, w, fit, torch.tensor(pi), 48))
        rk = lambda x: pd.Series(x).rank().values
        half = reorder(base, .5 * rk(base) + .5 * rk(g1), cls, ['person'])
        ref = reorder(half, .5 * rk(half) + .5 * rk(g2), cls, ['person'])
        rus = np.nan_to_num(q.id.map(oof.detector_pose_lr).values.astype(float), nan=.5)
        v = {'база (люди 0.866)': base, 'геометрия 140 (≈0.907)': half, 'рефайнер 280 (≈0.911)': ref,
             'Ruslan вместо': reorder(base, rus, cls, ['person'])}
        for wt in (.25, .5, .75):
            v[f'0.911 + Ruslan {wt}'] = reorder(ref, (1 - wt) * rk(ref) + wt * rk(rus), cls, ['person'])
        for k, s in v.items():
            res.setdefault(k, {})[f'{day} все'] = average_precision_score(q.error, s)
            res[k][f'{day} люди'] = average_precision_score(q.error[P], s[P])
        keep.append(q[['id', 'error']].assign(day=day, g140=g1, g280=g2, ruslan=rus))
    table = pd.DataFrame(res).T.round(4)
    print(table.to_string())
    table.to_csv(DATA / 'repro_refiner.csv')
    pd.concat(keep).to_csv(DATA / 'repro_refiner_oof.csv', index=False)


if __name__ == '__main__':
    main()
