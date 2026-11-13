"""Дообучение YOLO на ручной разметке train: детектор учится рисовать боксы «как разметчик»
(полный человек с ногами под столом, плотные кисти и т.п.). Затем его боксы — переразметка для reannotate.

Веса: yolo11n.pt, https://github.com/ultralytics/assets/releases (COCO, официальные ultralytics).
Прогоны (каждый сохраняется отдельно, уже готовые пропускаются):
  d25 — обучение на 25.10, предсказания на целевых кадрах 26.10;
  d26 — обучение на 26.10, предсказания на 25.10;
  all — обучение на всём train, предсказания на целевых кадрах test.
Результат: data/ft_<fold>.csv (frame_id, ext_class, ext_conf, x1..y2 в пикселях 1280x720).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from features import D

ROOT = Path(__file__).parent
WORK = ROOT / 'data' / 'yolo'
CLASSES = ['person', 'nude_hands', 'cup', 'tray']
EPOCHS, IMGSZ = int(os.environ.get('EPOCHS', 60)), int(os.environ.get('IMGSZ', 960))
BASE = os.environ.get('BASE', 'yolo11n')  # yolo11n / yolo11s; у не-n моделей суффикс в имени прогона
SEED = int(os.environ.get('SEED', 0))     # другой seed — ещё одна модель для ансамбля (суффикс _<seed>)
SUFFIX = (('' if BASE == 'yolo11n' else f'_{BASE[-1]}') + (f'_{SEED}' if SEED else '')
          + (f'_{IMGSZ}' if IMGSZ != 960 else ''))


def build_dataset(fold, frames):
    h = pd.read_csv(D / 'human_boxes_train.csv')
    h['class_name'] = h.class_name.replace(json.load(open(D / 'scoring_policy.json'))['aliases'])
    h = h[h.class_name.isin(CLASSES)]
    base = WORK / fold
    for sub in ('images', 'labels'):
        (base / sub).mkdir(parents=True, exist_ok=True)
    for f in frames:
        img = base / 'images' / f'{f}.jpg'
        if not img.exists():
            img.symlink_to((D / f'{f}.jpg').resolve())
        rows = h[h.frame_id == f]
        (base / 'labels' / f'{f}.txt').write_text(''.join(
            f'{CLASSES.index(r.class_name)} {r.cx:.6f} {r.cy:.6f} {r.w:.6f} {r.h:.6f}\n' for r in rows.itertuples()))
    cfg = base / 'data.yaml'
    cfg.write_text(f'path: {base}\ntrain: images\nval: images\nnames: {dict(enumerate(CLASSES))}\n')
    return cfg


def predict(model, frames, out):
    rows = []
    for f in frames:
        r = model.predict(str(D / f'{f}.jpg'), conf=0.05, iou=0.6, imgsz=IMGSZ, verbose=False, max_det=300)[0]
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            rows.append({'frame_id': f, 'ext_class': CLASSES[int(k)], 'ext_conf': c, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, len(rows), flush=True)


def main(folds):
    train_frames = pd.read_csv(D / 'train_context.csv').frame_id
    test_frames = pd.read_csv(D / 'test_context.csv').frame_id
    day = train_frames.str[-15:-7]
    plan = {'d25': (train_frames[day == '20261025'], train_frames[day == '20261026']),
            'd26': (train_frames[day == '20261026'], train_frames[day == '20261025']),
            'all': (train_frames, test_frames)}
    for fold in folds:
        out = ROOT / 'data' / f'ft_{fold}{SUFFIX}.csv'
        if out.exists():
            print('skip', fold, flush=True)
            continue
        fit_frames, pred_frames = plan[fold]
        cfg = build_dataset(fold, fit_frames)
        w = ROOT / 'weights' / f'{BASE}.pt'
        model = YOLO(w if w.exists() else f'{BASE}.pt')
        model.train(data=str(cfg), epochs=EPOCHS, imgsz=IMGSZ, batch=4, workers=2, device='cpu', cache=False,
                    project=str(WORK / 'runs'), name=f'{fold}{SUFFIX}', exist_ok=True, val=False, plots=False, verbose=False,
                    fliplr=0.5, mosaic=1.0, close_mosaic=10, patience=0, seed=SEED)
        predict(model, pred_frames, out)


if __name__ == '__main__':
    main(sys.argv[1:] or ['d25', 'd26', 'all'])
