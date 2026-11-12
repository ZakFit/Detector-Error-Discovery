"""Признаки второго мнения: согласие детекции с независимыми COCO-моделями (см. second_opinion.py)."""
from pathlib import Path

import numpy as np
import pandas as pd

from features import BOX, iou_matrix

ROOT = Path(__file__).parent
MATCH = {'person': {'person'}, 'cup': {'cup', 'wine glass', 'bowl', 'bottle'}, 'tray': {'dining table', 'bowl'}}


def load_opinions():
    seg = pd.read_csv(ROOT / 'data' / 'second_opinion_yolov8s-seg.csv')
    pose = pd.read_csv(ROOT / 'data' / 'second_opinion_yolo11m-pose.csv')
    return {f: g for f, g in seg.groupby('frame_id')}, {f: g for f, g in pose.groupby('frame_id')}


def wrist_points(pose_frame, min_conf=0.3):
    pts = []
    for side in ('lw', 'rw'):
        p = pose_frame[[f'{side}_x', f'{side}_y', f'{side}_c']].values
        pts.append(p[p[:, 2] >= min_conf])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def opinion_features(df):
    seg, pose = load_opinions()
    rows = []
    for fid, P in df.groupby('frame_id'):
        S = seg.get(fid, pd.DataFrame(columns=['ext_class', 'ext_conf'] + BOX))
        Q = pose.get(fid)
        for _, p in P.iterrows():
            box = p[BOX].values.astype(float)[None]
            f = {'id': p.id}
            # лучший IoU с моделью seg по сопоставимым классам, с уверенностью внешнего бокса
            cand = S[S.ext_class.isin(MATCH.get(p.class_name, set()))]
            iou = iou_matrix(box, cand[BOX].values.astype(float))[0] if len(cand) else np.zeros(0)
            f['seg_iou'] = iou.max() if len(iou) else 0.0
            f['seg_iou_conf'] = (iou * cand.ext_conf.values).max() if len(iou) else 0.0
            # любой класс seg, который сильно пересекается, но не сопоставим (подсказка ошибки класса)
            other = S[~S.ext_class.isin(MATCH.get(p.class_name, set()))]
            oi = iou_matrix(box, other[BOX].values.astype(float))[0] if len(other) else np.zeros(0)
            f['seg_other_iou'] = (oi * other.ext_conf.values).max() if len(oi) else 0.0
            if p.class_name == 'person' and Q is not None:
                pi = iou_matrix(box, Q[BOX].values.astype(float))[0]
                f['pose_iou'] = pi.max()
                f['pose_iou_conf'] = (pi * Q.ext_conf.values).max()
                j = pi.argmax()
                # насколько бокс позы выходит ниже нашего (обрезка по столу)
                h = box[0, 3] - box[0, 1]
                f['pose_dy2'] = (Q.y2.values[j] - box[0, 3]) / h if pi.max() > 0.3 else 0.0
            if p.class_name == 'nude_hands' and Q is not None:
                w = wrist_points(Q)
                x1, y1, x2, y2 = box[0]
                bw, bh = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if len(w):
                    d = np.hypot((w[:, 0] - cx) / bw, (w[:, 1] - cy) / bh)
                    f['wrist_dist'] = d.min()
                    inside = d < 1.0
                    f['wrist_conf_in'] = w[inside, 2].max() if inside.any() else 0.0
                else:
                    f['wrist_dist'], f['wrist_conf_in'] = 5.0, 0.0
            rows.append(f)
    return pd.DataFrame(rows)


OPINION_FEATURES = ['seg_iou', 'seg_iou_conf', 'seg_other_iou', 'pose_iou', 'pose_iou_conf', 'pose_dy2',
                    'wrist_dist', 'wrist_conf_in']
