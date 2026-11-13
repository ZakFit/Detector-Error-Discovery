"""Итог: переразметка открытыми моделями (reannotate.py) + переразметка YOLO, дообученными на ручной разметке
(finetune_yolo.py, прогон 'all'). score = среднее двух score; без подбора весов.

Согласие с дообученными YOLO = IoU лучшего бокса того же класса (с --conf: IoU × уверенность),
усреднённое по моделям-ансамблю.
Проверка (обучение YOLO11n на 26.10 → 25.10): AP 0.846 (reannotate) → 0.887 (IoU) → 0.894 (IoU × уверенность).
Public: `python final_ft.py ft_all.csv` (одна YOLO11n, голый IoU) — 0.842.
--match: сопоставление один-к-одному, как в метке (отложенный день: 0.902 / 0.876 вместе с --conf
против 0.887 / 0.837 у рецепта 0.842).

  python final_ft.py [--match] [--conf] [ft_all.csv ft_all_s.csv ...]   (по умолчанию — все data/ft_all*.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import BOX, D, iou_matrix, load
from reannotate import agreement, amodal_iou, score

ROOT = Path(__file__).parent


def ft_agree(df, path, min_conf=0.1, weighted=False):
    ft = pd.read_csv(path)
    frames = {k: g for k, g in ft[ft.ext_conf >= min_conf].groupby('frame_id')}
    out = []
    for _, p in df.iterrows():
        g = frames.get(p.frame_id)
        g = g[g.ext_class == p.class_name] if g is not None else None
        if g is None or not len(g):
            out.append(0.0)
            continue
        iou = iou_matrix(p[BOX].values.astype(float)[None], g[BOX].values)[0]
        out.append(float((iou * g.ext_conf.values).max() if weighted else iou.max()))
    return np.array(out)


def ft_agree_matched(df, path, min_conf=0.1, weighted=False):
    """Как в метке: на кадре и классе детекции сопоставляются с боксами YOLO один-к-одному — сначала максимум пар
    с IoU >= 0.6, затем максимум суммарного IoU. Согласие = IoU с доставшимся боксом (0 без пары), так что
    дубль на одном объекте получает низкое согласие."""
    from scipy.optimize import linear_sum_assignment
    ft = pd.read_csv(path)
    frames = {k: g for k, g in ft[ft.ext_conf >= min_conf].groupby('frame_id')}
    out = pd.Series(0.0, index=df.index)
    for (f, cls), P in df.groupby(['frame_id', 'class_name']):
        g = frames.get(f)
        g = g[g.ext_class == cls] if g is not None else None
        if g is None or not len(g):
            continue
        iou = iou_matrix(P[BOX].values.astype(float), g[BOX].values)
        rows, cols = linear_sum_assignment(-np.where(iou >= 0.6, 1e3 + iou, iou))
        val = iou[rows, cols] * (g.ext_conf.values[cols] if weighted else 1.0)
        out[P.index[rows]] = val
    return out.values


def main():
    args = sys.argv[1:]
    weighted, matched = '--conf' in args, '--match' in args
    agree_fn = ft_agree_matched if matched else ft_agree
    paths = [ROOT / 'data' / p for p in args if not p.startswith('--')] or sorted((ROOT / 'data').glob('ft_all*.csv'))
    te, *_ = load('test')
    te = te[te.class_name != 'table'].reset_index(drop=True)
    te['agree'], te['amodal'] = agreement(te), amodal_iou(te)
    te['ft'] = np.mean([agree_fn(te, p, weighted=weighted) for p in paths], axis=0)
    te['score'] = (score(te) + 1 - te.ft) / 2
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    sub = template[['id']].merge(te[['id', 'score']].astype({'id': str}), on='id', how='left', validate='one_to_one')
    assert len(sub) == len(template) and sub.score.notna().all() and sub.score.between(0, 1).all()
    tag = "_".join(p.stem.replace("ft_", "") for p in paths) + ('_match' if matched else '') + ('_conf' if weighted else '')
    out = ROOT / 'submissions' / f'sub_ft_{tag}.csv'
    sub.to_csv(out, index=False)
    print(out, 'models:', [p.name for p in paths])


if __name__ == '__main__':
    main()
