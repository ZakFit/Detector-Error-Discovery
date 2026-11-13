"""Предсказатель IoU с ручным боксом (идея IoU-Net, Jiang et al. 2018) поверх замороженных эмбеддингов.

Основа (backbone) прогоняется один раз на кадр → карта признаков; для каждого бокса RoIAlign по области
бокса (с небольшим контекстом) → вектор; голова — ридж-регрессия на IoU, отдельно по классам.
Обучение: сдвинутые/масштабированные копии ручных боксов (IoU известен точно) + реальные детекции дня обучения.
Основы: resnet18 (torchvision), dinov2 (facebook/dinov2-small), sam (facebook/sam-vit-base) — меняются флагом.

  python iou_net.py <backbone> → AUC по дням (обучение на одном дне, проверка на другом) + data/iou_<backbone>.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from torchvision.ops import roi_align

from features import BOX, D, iou_matrix, load

ROOT = Path(__file__).parent
CLASSES = ['person', 'nude_hands', 'cup', 'tray']
GRID, CONTEXT, N_JITTER = 4, 1.25, 40
torch.set_num_threads(int(__import__('os').environ.get('THREADS', 10)))


class Backbone:
    def __init__(self, name):
        self.name = name
        if name == 'resnet18':
            import torchvision
            m = torchvision.models.resnet18(weights='IMAGENET1K_V1').eval()
            self.net = torch.nn.Sequential(*list(m.children())[:-3])  # до layer3, stride 16
            self.size, self.stride = (1280, 720), 16
        elif name == 'dinov2':
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained('facebook/dinov2-small').eval()
            self.size, self.stride = (896, 504), 14
        elif name == 'sam':
            from transformers import SamModel
            self.net = SamModel.from_pretrained('facebook/sam-vit-base').eval()
            self.size, self.stride = (1024, 576), 16

    @torch.no_grad()
    def feature_map(self, path):
        img = Image.open(path).convert('RGB').resize(self.size, Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = (x - mean) / std
        if self.name == 'resnet18':
            return self.net(x)
        if self.name == 'dinov2':
            tok = self.net(pixel_values=x).last_hidden_state[:, 1:]
            h, w = self.size[1] // 14, self.size[0] // 14
            return tok.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        pad = torch.zeros(1, 3, 1024, 1024)
        pad[:, :, :x.shape[2], :x.shape[3]] = x
        return self.net.get_image_embeddings(pixel_values=pad)[:, :, :self.size[1] // 16 + 1]

    def roi(self, fmap, boxes):
        b = np.asarray(boxes, float).copy()
        cx, cy, w, h = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        b = np.c_[cx - CONTEXT * w / 2, cy - CONTEXT * h / 2, cx + CONTEXT * w / 2, cy + CONTEXT * h / 2]
        scale = self.size[0] / 1280 / self.stride
        rois = torch.tensor(np.c_[np.zeros(len(b)), b], dtype=torch.float32)
        f = roi_align(fmap, rois, output_size=GRID, spatial_scale=scale, sampling_ratio=2, aligned=True)
        return f.flatten(1).numpy().astype(np.float32)


def human_boxes():
    h = pd.read_csv(D / 'human_boxes_train.csv')
    h['class_name'] = h.class_name.replace(json.load(open(D / 'scoring_policy.json'))['aliases'])
    h = h[h.class_name.isin(CLASSES)].copy()
    h['x1'], h['y1'] = (h.cx - h.w / 2) * 1280, (h.cy - h.h / 2) * 720
    h['x2'], h['y2'] = (h.cx + h.w / 2) * 1280, (h.cy + h.h / 2) * 720
    return h


def jitter(g, rng):
    w, h = g[2] - g[0], g[3] - g[1]
    cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    sig = np.where(rng.random(N_JITTER) < 0.7, 0.12, 0.35)[:, None]
    d = rng.normal(0, 1, (N_JITTER, 4)) * sig
    ncx, ncy = cx + d[:, 0] * w, cy + d[:, 1] * h
    nw, nh = w * np.exp(d[:, 2]), h * np.exp(d[:, 3])
    return np.c_[ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2].clip([0, 0, 0, 0], [1280, 720, 1280, 720])


def build(backbone):
    """Признаки и целевой IoU для: синтетики вокруг ручных боксов train, реальных детекций train и test."""
    cache = ROOT / 'data' / f'iou_feats_{backbone}.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    bb, rng, hb = Backbone(backbone), np.random.default_rng(0), human_boxes()
    tr, *_ = load('train')
    te, *_ = load('test')
    dets = pd.concat([tr.assign(part='train'), te.assign(part='test')])
    dets = dets[dets.class_name.isin(CLASSES)]
    rows = []
    frames = sorted(set(dets.frame_id) | set(hb.frame_id))
    for i, f in enumerate(frames):
        fmap = bb.feature_map(D / f'{f}.jpg')
        G = hb[hb.frame_id == f]
        for cls in CLASSES:
            g = G[G.class_name == cls][BOX].values
            P = dets[(dets.frame_id == f) & (dets.class_name == cls)]
            boxes, kind, ids = [], [], []
            for gb in g:
                j = jitter(gb, rng)
                boxes += list(j); kind += ['synth'] * len(j); ids += [None] * len(j)
            boxes += list(P[BOX].values); kind += list(P.part); ids += list(P.id)
            if not boxes:
                continue
            boxes = np.asarray(boxes, float)
            target = iou_matrix(boxes, g).max(1) if len(g) else np.zeros(len(boxes))
            feats = bb.roi(fmap, boxes)
            for k in range(len(boxes)):
                rows.append({'frame_id': f, 'class_name': cls, 'kind': kind[k], 'id': ids[k],
                             'iou': target[k], 'feat': feats[k]})
        if i % 20 == 0:
            print(backbone, i, len(frames), flush=True)
    df = pd.DataFrame(rows)
    df['day'] = df.frame_id.str[-15:-7]
    df.to_pickle(cache)
    return df


def fit_predict(train_rows, pred_rows):
    out = pd.Series(np.nan, index=pred_rows.index)
    for cls in CLASSES:
        group = ['cup', 'tray'] if cls in ('cup', 'tray') else [cls]  # подносов мало — учим вместе с чашками
        a = train_rows[train_rows.class_name.isin(group)]
        b = pred_rows[pred_rows.class_name == cls]
        if len(b) == 0 or len(a) < 20:
            continue
        m = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(np.stack(a.feat), a.iou)
        out[b.index] = m.predict(np.stack(b.feat))
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'resnet18'
    df = build(name)
    tr, *_ = load('train')
    err = tr.set_index('id').error
    real = df[df.kind == 'train'].copy()
    real['error'] = real.id.map(err)
    for d in sorted(real.day.unique()):
        fit = df[(df.day != d) & (df.kind != 'test')]
        q = real[real.day == d].copy()
        q['pred'] = fit_predict(fit, q)
        res = {c: round(roc_auc_score(x.error, -x.pred), 3) for c, x in q.groupby('class_name') if x.error.nunique() > 1}
        print(name, 'held-out day', d, 'AUC by class', res,
              '| corr(pred, true IoU)', round(np.corrcoef(q.pred, q.iou)[0, 1], 3))
    # финальная модель на всём train → предсказания для test и out-of-day для train
    te_rows = df[df.kind == 'test'].copy()
    te_rows['pred'] = fit_predict(df[df.kind != 'test'], te_rows)
    tr_rows = []
    for d in sorted(real.day.unique()):
        q = real[real.day == d].copy()
        q['pred'] = fit_predict(df[(df.day != d) & (df.kind != 'test')], q)
        tr_rows.append(q)
    out = pd.concat([pd.concat(tr_rows)[['id', 'pred']], te_rows[['id', 'pred']]])
    out.to_csv(ROOT / 'data' / f'iou_{name}.csv', index=False)
    print('saved', ROOT / 'data' / f'iou_{name}.csv')


if __name__ == '__main__':
    main()
