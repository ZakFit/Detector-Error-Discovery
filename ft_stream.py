"""Дообученные YOLO (finetune_yolo.py) на всех кадрах с контекстом (целевые + соседние ±30 с).

Для каждой модели две выдачи:
  nms='std'   — обычная (conf>=0.05, NMS IoU 0.6): соседние кадры как ансамбль во времени;
  nms='dense' — почти без подавления (conf>=0.02, NMS IoU 0.95): разброс положений бокса,
                из которого оценивается P(IoU с ручным боксом >= 0.6).
Проверочные модели (d25, d26) — на кадрах train, финальные (all*) — на кадрах test.
Результат: data/ftS_<run>.csv (frame_id, nms, ext_class, ext_conf, x1..y2).
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from features import D
from finetune_yolo import CLASSES, IMGSZ

ROOT = Path(__file__).parent
RUNS = ROOT / 'data' / 'yolo' / 'runs'


def frames(part):
    return sorted(pd.read_csv(D / f'{part}_stream_predictions.csv').frame_id.unique())


def run(name):
    out = ROOT / 'data' / f'ftS_{name}.csv'
    if out.exists():
        print('skip', name, flush=True)
        return
    part = 'train' if name.startswith('d2') else 'test'
    model = YOLO(RUNS / name / 'weights' / 'last.pt')
    rows = []
    for f in frames(part):
        for nms, conf, iou in (('std', 0.05, 0.6), ('dense', 0.02, 0.95)):
            r = model.predict(str(D / f'{f}.jpg'), conf=conf, iou=iou, imgsz=IMGSZ, verbose=False, max_det=500)[0]
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                rows.append({'frame_id': f, 'nms': nms, 'ext_class': CLASSES[int(k)], 'ext_conf': c,
                             'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, len(rows), flush=True)


if __name__ == '__main__':
    for name in sys.argv[1:] or ['d26', 'd25', 'all', 'all_s']:
        run(name)
