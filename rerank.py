"""Детерминированное переранжирование: 1-confidence + согласие с независимыми COCO-моделями.

Внутри каждого класса порядок задаётся суммой рангов (1-conf) и w*(1-согласие),
а сами значения score остаются исходным набором 1-conf класса — межклассовый порядок как у бейзлайна.
Согласие: cup и person — IoU*conf с боксом yolov8s-seg, nude_hands — уверенность запястья yolo11m-pose в боксе.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from features import D, build
from opinion_features import opinion_features

ROOT = Path(__file__).parent
W = 1.0
AGREE = {'cup': 'seg_iou_conf', 'person': 'seg_iou_conf', 'nude_hands': 'wrist_conf_in'}


def rerank(df, w=W):
    base = 1 - df.confidence.values
    out = base.copy()
    for cls, col in AGREE.items():
        idx = np.where((df.class_name == cls).values)[0]
        if len(idx) < 2:
            continue
        a = df[col].values[idx].astype(float)
        a = np.where(np.isnan(a), np.nanmedian(a) if np.isfinite(a).any() else 0, a)
        order = np.argsort(rankdata(base[idx]) + w * rankdata(1 - a), kind='stable')
        out[idx[order]] = np.sort(base[idx])
    return out


def main():
    w = float(sys.argv[1]) if len(sys.argv) > 1 else W
    tr = build('train').reset_index(drop=True)
    tr = tr.merge(opinion_features(tr), on='id', how='left')
    for day, p in tr.groupby('day'):
        print(day, 'baseline', round(average_precision_score(p.error, 1 - p.confidence), 4),
              'rerank', round(average_precision_score(p.error, rerank(p, w)), 4))

    te = build('test').reset_index(drop=True)
    te = te.merge(opinion_features(te), on='id', how='left')
    te['score'] = rerank(te, w)
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    sub = template[['id']].merge(te[['id', 'score']], on='id', how='left', validate='one_to_one')
    assert sub.score.notna().all() and sub.score.between(0, 1).all()
    out = ROOT / 'submissions' / f'sub_rerank_w{w:g}.csv'
    sub.to_csv(out, index=False)
    print(out, len(sub), 'spearman vs 1-conf:',
          round(sub.merge(te[['id', 'confidence']], on='id').pipe(lambda d: d.score.corr(1 - d.confidence, method='spearman')), 3))


if __name__ == '__main__':
    main()
