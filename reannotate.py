"""Переразметка независимыми моделями вместо confidence детектора.

Метка: детекция верна, если IoU с ручным боксом >= 0.6. Эмулируем ручную разметку открытыми моделями
и оцениваем каждую детекцию только по IoU с их боксами:
  руки, чашки, подносы — OWLv2 на увеличенном окне (owl_zoom.py);
  люди — консенсус OWLv2 (полный кадр), yolov8s-seg и yolo11m-pose.
score = 1 - согласие (+ крошечная добавка 1-confidence только для разрешения ничьих).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import BOX, D, iou_matrix, load

ROOT = Path(__file__).parent
ZOOM_MIN_SCORE = 0.1
PERSON_SOURCES = {  # файл, фильтр класса, порог уверенности
    'owl': ('second_opinion_owlv2.csv', 'a person', 0.1),
    'seg': ('second_opinion_yolov8s-seg.csv', 'person', 0.25),
    'pose': ('second_opinion_yolo11m-pose.csv', 'person', 0.25),
}


def best_iou(p, boxes):
    return float(iou_matrix(p[BOX].values.astype(float)[None], boxes[BOX].values)[0].max()) if len(boxes) else 0.0


def agreement(df, person_mode='mean'):
    z = pd.read_csv(ROOT / 'data' / 'owl_zoom.csv')
    zoom = {k: g for k, g in z[z.score >= ZOOM_MIN_SCORE].groupby('id')}
    src = {}
    for name, (fname, cls, thr) in PERSON_SOURCES.items():
        s = pd.read_csv(ROOT / 'data' / fname)
        src[name] = {k: g for k, g in s[(s.ext_class == cls) & (s.ext_conf >= thr)].groupby('frame_id')}
    out = []
    for _, p in df.iterrows():
        if p.class_name == 'person':
            ious = [best_iou(p, src[n].get(p.frame_id, pd.DataFrame(columns=BOX))) for n in src]
            out.append(np.mean(ious) if person_mode == 'mean' else np.median(ious))
        else:
            out.append(best_iou(p, zoom.get(p.id, pd.DataFrame(columns=BOX))))
    return np.array(out)


def amodal_iou(df):
    """Люди: IoU с «полным» боксом тела — бокс pose-модели, продлённый до всех ключевых точек
    (бёдра, колени, лодыжки, в т.ч. скрытые столом). Моделирует правило разметки «всё тело»."""
    pose = pd.read_csv(ROOT / 'data' / 'second_opinion_yolo11m-pose.csv')
    frames = {k: g for k, g in pose[pose.ext_conf >= 0.25].groupby('frame_id')}
    out = []
    for _, p in df.iterrows():
        g = frames.get(p.frame_id)
        if p.class_name != 'person' or g is None:
            out.append(0.0)
            continue
        pb = p[BOX].values.astype(float)[None]
        iou = iou_matrix(pb, g[BOX].values)[0]
        if iou.max() < 0.1:
            out.append(0.0)
            continue
        q = g.iloc[iou.argmax()]
        xs = [q[f'k{i}_x'] for i in range(17)]
        full = [[min([q.x1] + xs), q.y1, max([q.x2] + xs), max([q.y2] + [q[f'k{i}_y'] for i in range(11, 17)])]]
        out.append(float(iou_matrix(pb, np.array(full))[0, 0]))
    return np.array(out)


def person_agree(df):
    """Люди: среднее трёх независимых сигналов — согласие моделей, полный бокс тела, confidence."""
    return np.where(df.class_name == 'person', (df.agree + df.amodal + df.confidence) / 3, df.agree)


def score(df):
    return np.clip(1 - person_agree(df) + 0.01 * (1 - df.confidence.values), 0, 1)


def main():
    tr, *_ = load('train')
    tr = tr[tr.class_name != 'table'].reset_index(drop=True)
    tr['day'] = tr.frame_id.str[-15:-7]
    mode = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    tr['agree'], tr['amodal'] = agreement(tr, mode), amodal_iou(tr)
    print(mode, {d: round(average_precision_score(q.error, score(q)), 4) for d, q in tr.groupby('day')},
          '| baseline', {d: round(average_precision_score(q.error, 1 - q.confidence), 4) for d, q in tr.groupby('day')})
    te, *_ = load('test')
    te = te[te.class_name != 'table'].reset_index(drop=True)
    te['agree'], te['amodal'] = agreement(te, mode), amodal_iou(te)
    te['score'] = score(te)
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    sub = template[['id']].merge(te[['id', 'score']].astype({'id': str}), on='id', how='left', validate='one_to_one')
    assert len(sub) == len(template) and sub.score.notna().all()
    out = ROOT / 'submissions' / 'sub_reannotate_person3.csv'
    sub.to_csv(out, index=False)
    print(out)


if __name__ == '__main__':
    main()
