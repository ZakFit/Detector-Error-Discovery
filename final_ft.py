"""Итог: переразметка открытыми моделями (reannotate.py) + переразметка YOLO, дообученным на ручной разметке
(finetune_yolo.py, прогон 'all'). score = среднее двух score; без подбора весов.

Проверка (обучение YOLO на 26.10 → 25.10): AP 0.846 (reannotate) → 0.887 (смесь).
"""
from pathlib import Path

import pandas as pd

from eval_finetune import ft_iou
from features import D, load
from reannotate import agreement, amodal_iou, score

ROOT = Path(__file__).parent


def main():
    te, *_ = load('test')
    te = te[te.class_name != 'table'].reset_index(drop=True)
    te['agree'], te['amodal'] = agreement(te), amodal_iou(te)
    te['ft'] = ft_iou(te, ROOT / 'data' / 'ft_all.csv', 0.1)
    te['score'] = (score(te) + 1 - te.ft) / 2
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    sub = template[['id']].merge(te[['id', 'score']].astype({'id': str}), on='id', how='left', validate='one_to_one')
    assert len(sub) == len(template) and sub.score.notna().all() and sub.score.between(0, 1).all()
    out = ROOT / 'submissions' / 'sub_reannotate_ft.csv'
    sub.to_csv(out, index=False)
    print(out, 'ft coverage (IoU>0):', round((te.ft > 0).mean(), 2))


if __name__ == '__main__':
    main()
