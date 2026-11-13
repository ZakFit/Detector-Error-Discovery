"""Итог: переразметка открытыми моделями (reannotate.py) + переразметка YOLO, дообученными на ручной разметке
(finetune_yolo.py, прогон 'all'). score = среднее двух score; без подбора весов.

Согласие с дообученными YOLO = IoU лучшего бокса того же класса (с --conf: IoU × уверенность),
усреднённое по моделям-ансамблю.
Проверка (обучение YOLO11n на 26.10 → 25.10): AP 0.846 (reannotate) → 0.887 (IoU) → 0.894 (IoU × уверенность).
Public: `python final_ft.py ft_all.csv` (одна YOLO11n, голый IoU) — 0.842.

  python final_ft.py [--conf] [ft_all.csv ft_all_s.csv ...]   (по умолчанию — все data/ft_all*.csv)
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


def main():
    args = sys.argv[1:]
    weighted = '--conf' in args
    paths = [ROOT / 'data' / p for p in args if p != '--conf'] or sorted((ROOT / 'data').glob('ft_all*.csv'))
    te, *_ = load('test')
    te = te[te.class_name != 'table'].reset_index(drop=True)
    te['agree'], te['amodal'] = agreement(te), amodal_iou(te)
    te['ft'] = np.mean([ft_agree(te, p, weighted=weighted) for p in paths], axis=0)
    te['score'] = (score(te) + 1 - te.ft) / 2
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    sub = template[['id']].merge(te[['id', 'score']].astype({'id': str}), on='id', how='left', validate='one_to_one')
    assert len(sub) == len(template) and sub.score.notna().all() and sub.score.between(0, 1).all()
    tag = "_".join(p.stem.replace("ft_", "") for p in paths) + ('_conf' if weighted else '')
    out = ROOT / 'submissions' / f'sub_ft_{tag}.csv'
    sub.to_csv(out, index=False)
    print(out, 'models:', [p.name for p in paths])


if __name__ == '__main__':
    main()
