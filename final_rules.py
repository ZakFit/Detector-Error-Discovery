"""Два непроверенных правила поверх 0.911 (последний день, «рандомные» идеи; параметры заданы заранее, не подбирались).

1. Люди — видны ноги ниже бокса. Берём позу YOLO11m-pose с наибольшим IoU с боксом (≥ 0.3). Если колено или
   лодыжка (k13–k16, conf ≥ 0.5) лежит ниже нижней границы бокса больше чем на 5 % его высоты — бокс обрезан,
   а ручной бокс person охватывает всё тело → вероятна ошибка геометрии.
2. Чашки — не на столе. Если центр чашки не попадает ни в один бокс table того же кадра (детекции потока,
   столы почти всегда верны) и рядом нет руки — вероятно ложное срабатывание (телефон, вещь).

Флаг поднимает бокс внутри класса: ключ = доля-ранг базы + 0.25·флаг; значения score класса берутся из базы,
поэтому межклассовый порядок не меняется.
  python final_rules.py <base.csv>   → submissions/<base>_rules.csv  (и печать доли ошибок у флагов на train)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import BOX, D, iou_matrix, load

ROOT = Path(__file__).parent
W = 0.25


def legs_below(det, pose):
    flag = np.zeros(len(det))
    for i, r in enumerate(det.itertuples()):
        if r.class_name != 'person':
            continue
        g = pose[pose.frame_id == r.frame_id]
        if not len(g):
            continue
        m = iou_matrix(np.array([[r.x1, r.y1, r.x2, r.y2]], float), g[BOX].values)[0]
        if m.max() < .3:
            continue
        p = g.iloc[m.argmax()]
        h = r.y2 - r.y1
        flag[i] = any(p[f'k{k}_c'] >= .5 and p[f'k{k}_y'] > r.y2 + .05 * h for k in (13, 14, 15, 16))
    return flag


def cup_off_table(det, stream):
    flag = np.zeros(len(det))
    for i, r in enumerate(det.itertuples()):
        if r.class_name != 'cup':
            continue
        g = stream[stream.frame_id == r.frame_id]
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        inside = lambda b: ((b.x1 <= cx) & (cx <= b.x2) & (b.y1 <= cy) & (cy <= b.y2)).any()
        hands = g[g.class_name == 'nude_hands']
        near_hand = len(hands) and iou_matrix(np.array([[r.x1, r.y1, r.x2, r.y2]], float), hands[BOX].values)[0].max() > 0
        flag[i] = not inside(g[g.class_name == 'table']) and not near_hand
    return flag


def flags(part):
    det, stream, *_ = load(part)
    det = det[det.class_name != 'table'].reset_index(drop=True)
    pose = pd.read_csv(ROOT / 'data' / 'second_opinion_yolo11m-pose.csv')
    return det, legs_below(det, pose), cup_off_table(det, stream)


def main():
    tr, l, c = flags('train')
    for name, f in (('ноги ниже бокса (люди)', l), ('чашка не на столе', c)):
        cls = 'person' if 'люди' in name else 'cup'
        m = tr.class_name == cls
        print(f'train {name}: флагов {int(f.sum())}/{m.sum()}, ошибок у флагов {tr.error[f > 0].mean():.2f}, '
              f'в классе {tr.error[m].mean():.2f}')
    base_path = Path(sys.argv[1])
    base = pd.read_csv(base_path, dtype={'id': str})
    te, l, c = flags('test')
    flag = pd.Series(l + c, index=te.id)
    out = base.copy()
    cls = base.id.map(te.set_index('id').class_name)
    for k in ('person', 'cup'):
        idx = np.flatnonzero(cls == k)
        s = base.score.values[idx]
        key = pd.Series(s).rank(pct=True).values + W * flag.reindex(base.id.values[idx]).values
        out.loc[out.index[idx[np.argsort(key, kind='stable')]], 'score'] = np.sort(s)
        print(f'test {k}: флагов {int(flag.reindex(base.id.values[idx]).sum())}/{len(idx)}')
    dst = ROOT / 'submissions' / f'{base_path.stem}_rules.csv'
    out.to_csv(dst, index=False)
    print(dst, '| строк изменено:', int((out.score != base.score).sum()))


if __name__ == '__main__':
    main()
