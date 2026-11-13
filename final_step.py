"""Итог 13.11: рецепт 0.842 + пороговое согласие с дообученными YOLO во времени — для рук, чашек, подносов.

Метка — ступенька: детекция верна, если найдётся ручной бокс с IoU >= 0.6. Поэтому вместо сырого IoU
спрашиваем то же, что метка: есть ли у дообученной YOLO уверенный бокс с IoU >= 0.6 (выдача почти без NMS,
т.е. с учётом её неуверенности в положении бокса), и усредняем это по целевому и соседним кадрам (±30 с).
Сигнал усредняется по моделям ансамбля (data/ftS_<run>.csv из ft_stream.py).

Люди не трогаются (score и порядок как в 0.842). Руки/чашки/подносы переупорядочиваются сигналом,
значения score класса — из 0.842, так что порядок между классами не меняется.
Проверка (temporal_eval.py, люди как в базе): AP 25.10 0.889 → 0.902, 26.10 0.829 → 0.855.

  python final_step.py [all all_s all_1 ...]   (по умолчанию — все data/ftS_all*.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import D, load
from final_ft import ft_agree
from reannotate import agreement, amodal_iou, score
from temporal_eval import REORDERED, signals

ROOT = Path(__file__).parent


def main():
    runs = sys.argv[1:] or sorted(p.stem.replace('ftS_', '') for p in (ROOT / 'data').glob('ftS_all*.csv'))
    te, *_ = load('test')
    ctx = pd.read_csv(D / 'test_context.csv')
    te = te[te.class_name != 'table'].reset_index(drop=True)
    te['agree'], te['amodal'] = agreement(te), amodal_iou(te)
    re_score = score(te)
    base = (re_score + 1 - ft_agree(te, ROOT / 'data' / 'ft_all.csv')) / 2          # рецепт 0.842
    step_t = np.mean([signals(te, pd.read_csv(ROOT / 'data' / f'ftS_{r}.csv'), ctx).step_t.values
                      for r in runs], axis=0)
    ens = (re_score + 1 - step_t) / 2
    out = base.copy()
    for cls in REORDERED:
        idx = np.where((te.class_name == cls).values)[0]
        out[idx[np.argsort(ens[idx], kind='stable')]] = np.sort(base[idx])
    te['score'] = out
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    sub = template[['id']].merge(te[['id', 'score']].astype({'id': str}), on='id', how='left', validate='one_to_one')
    assert len(sub) == len(template) and sub.score.notna().all() and sub.score.between(0, 1).all()
    out_path = ROOT / 'submissions' / f'sub_step_{"_".join(runs)}.csv'
    sub.to_csv(out_path, index=False)
    print(out_path, 'models:', runs)


if __name__ == '__main__':
    main()
