"""Сигналы согласия с дообученной YOLO для рук/чашек/подносов и их проверка на отложенных днях.

  iou      — IoU с лучшим боксом того же класса на целевом кадре (рецепт 0.842);
  time     — среднее этого IoU по целевому и соседним кадрам (±30 с): ансамбль во времени;
  step     — пороговая версия: макс. уверенность YOLO среди её боксов с IoU >= 0.6 (метка — ступенька на 0.6);
  step_t   — step, усреднённый по времени.
Проверка в форме сабмита: люди как в рецепте 0.842; руки/чашки/подносы переупорядочены сигналом,
значения score класса — из базы.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import BOX, D, iou_matrix, load

ROOT = Path(__file__).parent
REORDERED = ['nude_hands', 'cup', 'tray']


def signals(df, ftS, ctx):
    """df — детекции; ftS — выдача ft_stream.py одной модели; ctx — контекст кадров (frame_id, previous, next)."""
    std = {k: g for k, g in ftS[ftS.nms == 'std'].groupby('frame_id')}
    dense = {k: g for k, g in ftS[ftS.nms == 'dense'].groupby('frame_id')}
    nb = ctx.set_index('frame_id')
    out = {k: [] for k in ('iou', 'time', 'step', 'step_t', 'n_frames')}
    for _, p in df.iterrows():
        box = p[BOX].values.astype(float)[None]
        frames = [p.frame_id] + [f for f in (nb.previous.get(p.frame_id), nb.next.get(p.frame_id))
                                 if isinstance(f, str) and f in std]
        ious, steps = [], []
        for f in frames:
            g = std.get(f, ftS.iloc[:0])
            g = g[g.ext_class == p.class_name]
            ious.append(float(iou_matrix(box, g[BOX].values)[0].max()) if len(g) else 0.0)
            h = dense.get(f)
            h = h[h.ext_class == p.class_name] if h is not None else h
            if h is None or not len(h):
                steps.append(0.0)
                continue
            i = iou_matrix(box, h[BOX].values)[0]
            steps.append(float(h.ext_conf.values[i >= 0.6].max()) if (i >= 0.6).any() else 0.0)
        out['iou'].append(ious[0])
        out['time'].append(np.mean(ious))
        out['step'].append(steps[0])
        out['step_t'].append(np.mean(steps))
        out['n_frames'].append(len(frames))
    return pd.DataFrame(out, index=df.index)


def reorder(base, signal, classes):
    """Внутри классов — порядок по (1 − signal), значения score — из base."""
    out = base.copy()
    for c in REORDERED:
        idx = np.where(classes == c)[0]
        out[idx[np.argsort(-signal[idx], kind='stable')]] = np.sort(base[idx])
    return out


def main():
    from reannotate import agreement, amodal_iou, score
    tr, *_ = load('train')
    ctx = pd.read_csv(D / 'train_context.csv')
    tr = tr[tr.class_name != 'table'].reset_index(drop=True)
    tr['day'] = tr.frame_id.str[-15:-7]
    tr['agree'], tr['amodal'] = agreement(tr), amodal_iou(tr)
    for day, fold in (('20261025', 'd26'), ('20261026', 'd25')):
        q = tr[tr.day == day].reset_index(drop=True)
        s = signals(q, pd.read_csv(ROOT / 'data' / f'ftS_{fold}.csv'), ctx)
        re_score = score(q)
        base = (re_score + 1 - s.iou.values) / 2
        cls = q.class_name.values
        nonp = q.class_name.isin(REORDERED)
        print(f'held-out {day} (YOLO {fold}); соседних кадров в среднем {s.n_frames.mean():.2f}')
        print(f'  base (рецепт 0.842)       AP {average_precision_score(q.error, base):.4f}')
        for name in ('time', 'step', 'step_t'):
            sig = s[name].values
            # чем выше согласие, тем вероятнее детекция верна: передаём ens-score = (re + 1 - sig)/2
            ens = (re_score + 1 - sig) / 2
            sc = reorder(base, -ens, cls)
            auc = {c: round(roc_auc_score(x.error, ens[x.index]), 3) for c, x in q[nonp].groupby('class_name')
                   if x.error.nunique() > 1}
            print(f'  {name:7s} → руки/чашки/подносы  AP {average_precision_score(q.error, sc):.4f}  AUC {auc}')
        auc0 = {c: round(roc_auc_score(x.error, base[x.index]), 3) for c, x in q[nonp].groupby('class_name')
                if x.error.nunique() > 1}
        print(f'  (base AUC {auc0})')


if __name__ == '__main__':
    main()
