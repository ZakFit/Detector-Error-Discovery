"""Второе мнение: независимые открытые детекторы на целевых кадрах.

Веса — официальные релизы ultralytics (https://github.com/ultralytics/assets/releases):
  yolov8s-seg.pt  — COCO, 80 классов (person, cup, wine glass, bowl, ...)
  yolo11m-pose.pt — COCO-pose: person + 17 ключевых точек (запястья 9, 10)
Результат: data/second_opinion_<model>.csv с боксами в пикселях 1280x720.
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from features import D

ROOT = Path(__file__).parent
WRISTS = {'lw': 9, 'rw': 10, 'le': 7, 're': 8}   # запястья и локти
# все 17 точек COCO: k0..k16 (0 нос, 5-6 плечи, 11-12 бёдра, 13-14 колени, 15-16 лодыжки)
ALL_KP = {f'k{i}': i for i in range(17)}


def run(weights):
    frames = pd.concat([pd.read_csv(D / f'{p}_context.csv').frame_id for p in ('train', 'test')]).unique()
    model = YOLO(weights)
    rows = []
    for i, fid in enumerate(frames):
        r = model.predict(str(D / f'{fid}.jpg'), conf=0.05, iou=0.6, imgsz=1280, verbose=False, max_det=300)[0]
        kp = r.keypoints.data.tolist() if r.keypoints is not None else None
        for j, ((x1, y1, x2, y2), c, k) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                                         r.boxes.cls.tolist())):
            row = {'frame_id': fid, 'ext_class': r.names[int(k)], 'ext_conf': c, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            if kp is not None:
                for name, idx in {**WRISTS, **ALL_KP}.items():
                    row[f'{name}_x'], row[f'{name}_y'], row[f'{name}_c'] = kp[j][idx]
            rows.append(row)
        if i % 50 == 0:
            print(Path(weights).name, i, len(frames), flush=True)
    out = ROOT / 'data' / f'second_opinion_{Path(weights).stem}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, len(rows))


if __name__ == '__main__':
    for w in sys.argv[1:]:
        run(w)
