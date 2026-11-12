"""Итоговое решение: простая логика, три независимых сигнала, без обучения.

1. Базовый score = 1 - (confidence + согласие OWLv2) / 2.
   Согласие = max IoU * score бокса OWLv2 того же класса (запросы: hand, cup/glass, tray, person).
2. Внутри класса nude_hands порядок уточняется: средний ранг (базовый score, мало кожи в боксе,
   нет запястья pose-модели в боксе, нет совпадения с OWLv2 на увеличенном окне).
   Значения score класса сохраняются, меняется только порядок.

Требует: data/second_opinion_owlv2.csv (owl_opinion.py), data/second_opinion_yolo11m-pose.csv (second_opinion.py),
data/owl_zoom.csv (owl_zoom.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from features import BOX, D, iou_matrix, load
from opinion_features import wrist_points

ROOT = Path(__file__).parent
OWL_QUERIES = {'nude_hands': ['a human hand'], 'cup': ['a cup', 'a glass'], 'tray': ['a tray'], 'person': ['a person']}


def owl_agree(df):
    owl = pd.read_csv(ROOT / 'data' / 'second_opinion_owlv2.csv')
    frames = {k: g for k, g in owl.groupby('frame_id')}
    out = []
    for _, p in df.iterrows():
        o = frames.get(p.frame_id)
        same = o[o.ext_class.isin(OWL_QUERIES[p.class_name])] if o is not None else None
        if same is None or same.empty:
            out.append(0.0)
            continue
        iou = iou_matrix(p[BOX].values.astype(float)[None], same[BOX].values)[0]
        out.append(float((iou * same.ext_conf.values).max()))
    return np.array(out)


def skin_fraction(df):
    """Доля пикселей цвета кожи (YCrCb) в боксе."""
    cache, out = {}, []
    for _, p in df.iterrows():
        if p.frame_id not in cache:
            cache[p.frame_id] = np.asarray(Image.open(D / f'{p.frame_id}.jpg').convert('YCbCr')).astype(int)
        x1, y1, x2, y2 = [int(round(v)) for v in p[BOX]]
        c = cache[p.frame_id][max(y1, 0):y2, max(x1, 0):x2]
        cb, cr = c[..., 1], c[..., 2]
        out.append(float(((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).mean()))
    return np.array(out)


def wrist_conf(df):
    """Максимальная уверенность запястья yolo11m-pose внутри бокса (расстояние до центра < половины бокса)."""
    pose = pd.read_csv(ROOT / 'data' / 'second_opinion_yolo11m-pose.csv')
    frames = {k: g for k, g in pose.groupby('frame_id')}
    out = []
    for _, p in df.iterrows():
        q = frames.get(p.frame_id)
        w = wrist_points(q) if q is not None else np.zeros((0, 3))
        if not len(w):
            out.append(0.0)
            continue
        cx, cy, bw, bh = (p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2, p.x2 - p.x1, p.y2 - p.y1
        d = np.hypot((w[:, 0] - cx) / bw, (w[:, 1] - cy) / bh)
        out.append(float(w[d < 1.0, 2].max()) if (d < 1.0).any() else 0.0)
    return np.array(out)


def zoom_iou(df, min_score=0.1):
    """IoU детекции с лучшим боксом OWLv2 на увеличенном окне (owl_zoom.py); NaN, если окно не посчитано."""
    path = ROOT / 'data' / 'owl_zoom.csv'
    z = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['id', 'score'] + BOX)
    groups = {k: g for k, g in z[z.score >= min_score].groupby('id')}
    done = set(z.id)
    out = []
    for _, p in df.iterrows():
        g = groups.get(p.id)
        if g is not None:
            out.append(float(iou_matrix(p[BOX].values.astype(float)[None], g[BOX].values)[0].max()))
        else:
            out.append(0.0 if p.id in done else np.nan)
    return np.array(out)


def r01(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


# сигналы для перестановки внутри класса (чем больше значение, тем вероятнее детекция верна)
REORDER = {'nude_hands': ['skin', 'wrist', 'zoom'], 'cup': ['zoom'], 'tray': ['zoom']}


def score(df, reorder=REORDER):
    s = 1 - (df.confidence.values + df.owl.values) / 2
    out = s.copy()
    for cls, sigs in reorder.items():
        idx = np.where((df.class_name == cls).values)[0]
        if len(idx) < 2:
            continue
        ranks = [r01(s[idx])]
        for c in sigs:
            v = df[c].values[idx].astype(float)
            ranks.append(r01(-np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(v).any() else 0, v)))
        comb = np.mean(ranks, axis=0)
        out[idx[np.argsort(comb, kind='stable')]] = np.sort(s[idx])
    return out


def prepare(part):
    t, *_ = load(part)
    t = t[t.class_name.isin(OWL_QUERIES)].reset_index(drop=True)
    t['owl'], t['skin'], t['wrist'], t['zoom'] = owl_agree(t), skin_fraction(t), wrist_conf(t), zoom_iou(t)
    return t


def main():
    tr = prepare('train')
    tr['day'] = tr.frame_id.str[-15:-7]
    for day, q in tr.groupby('day'):
        print(day, 'baseline', round(average_precision_score(q.error, 1 - q.confidence), 4),
              'owl_avg', round(average_precision_score(q.error, 1 - (q.confidence + q.owl) / 2), 4),
              'final', round(average_precision_score(q.error, score(q)), 4))
    te = prepare('test')
    print('zoom computed for test hands:', te[te.class_name == 'nude_hands'].zoom.notna().mean().round(2))
    te['score'] = score(te)
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    sub = template[['id']].merge(te[['id', 'score']].astype({'id': str}), on='id', how='left', validate='one_to_one')
    assert len(sub) == len(template) and sub.score.notna().all() and sub.score.between(0, 1).all()
    out = ROOT / 'submissions' / 'sub_final_zoom.csv'
    sub.to_csv(out, index=False)
    print(out, len(sub))


if __name__ == '__main__':
    main()
