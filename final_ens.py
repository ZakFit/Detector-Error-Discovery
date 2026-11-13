"""Кандидат поверх 0.842: люди не трогаются, руки/чашки/подносы переупорядочиваются ансамблем дообученных YOLO.

- База — рецепт 0.842 (`final_ft.py ft_all.csv`): score = среднее (переразметка, 1 − IoU с YOLO11n).
- Люди: score и порядок ровно как в базе.
- Руки, чашки, подносы: порядок внутри класса — по той же формуле, но согласие = IoU × уверенность, усреднённое
  по всем дообученным моделям (data/ft_all*.csv: YOLO11n, YOLO11s, YOLO11n с другим seed);
  значения score класса берутся из базы, поэтому масштаб и порядок между классами не меняются.

Почему так: в командном сабмите 0.866 люди совпадают с 0.842, а порядок рук/чашек/подносов ближе всего
к ансамблю n+s (spearman 0.96–0.97) — ансамбль помогает этим классам на Public.

  python final_ens.py [ft_all.csv ft_all_s.csv ...]   (по умолчанию — все data/ft_all*.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import D, load
from final_ft import ft_agree
from reannotate import agreement, amodal_iou, score

ROOT = Path(__file__).parent
REORDERED = ['nude_hands', 'cup', 'tray']


def main():
    paths = [ROOT / 'data' / p for p in sys.argv[1:]] or sorted((ROOT / 'data').glob('ft_all*.csv'))
    te, *_ = load('test')
    te = te[te.class_name != 'table'].reset_index(drop=True)
    te['agree'], te['amodal'] = agreement(te), amodal_iou(te)
    re_score = score(te)
    base = (re_score + 1 - ft_agree(te, ROOT / 'data' / 'ft_all.csv')) / 2          # рецепт 0.842
    ens = (re_score + 1 - np.mean([ft_agree(te, p, weighted=True) for p in paths], axis=0)) / 2
    out = base.copy()
    for cls in REORDERED:
        idx = np.where((te.class_name == cls).values)[0]
        out[idx[np.argsort(ens[idx], kind='stable')]] = np.sort(base[idx])
    te['score'] = out
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    sub = template[['id']].merge(te[['id', 'score']].astype({'id': str}), on='id', how='left', validate='one_to_one')
    assert len(sub) == len(template) and sub.score.notna().all() and sub.score.between(0, 1).all()
    tag = '_'.join(p.stem.replace('ft_', '') for p in paths)
    out_path = ROOT / 'submissions' / f'sub_ens_{tag}.csv'
    sub.to_csv(out_path, index=False)
    print(out_path, 'models:', [p.name for p in paths])


if __name__ == '__main__':
    main()
