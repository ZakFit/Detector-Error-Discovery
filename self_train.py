"""Самообучение (test-time adaptation): дообученная YOLO-учитель размечает неразмеченные кадры целевой части
(целевые + соседние), уверенные боксы добавляются к ручной разметке, новая YOLO учится на всём вместе.
Цель — подстроить имитатор разметчика под условия целевых кадров (test — обед, train — в основном утро).

  проверка: python self_train.py check   — учитель d26 (ручная разметка 26.10) + псевдоразметка всех кадров 25.10
                                           → data/ft_d26pl.csv (предсказания на целевых кадрах 25.10)
  финал:    python self_train.py final   — учитель all (весь train) + псевдоразметка всех кадров test
                                           → data/ft_allpl.csv (предсказания на целевых кадрах test)
Порог псевдоразметки CONF (по умолчанию 0.5), эпохи EPOCHS (по умолчанию 40).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from features import D
from finetune_yolo import CLASSES, IMGSZ, WORK, predict

ROOT = Path(__file__).parent
CONF, EPOCHS = float(os.environ.get('CONF', 0.5)), int(os.environ.get('EPOCHS', 40))


def human():
    h = pd.read_csv(D / 'human_boxes_train.csv')
    h['class_name'] = h.class_name.replace(json.load(open(D / 'scoring_policy.json'))['aliases'])
    return h[h.class_name.isin(CLASSES)]


def write_label(path, rows):
    path.write_text(''.join(f'{CLASSES.index(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n' for c, cx, cy, w, h in rows))


def build(fold, labeled_frames, pseudo_frames, teacher):
    base = WORK / fold
    for sub in ('images', 'labels'):
        (base / sub).mkdir(parents=True, exist_ok=True)
    h, t = human(), teacher[(teacher.nms == 'std') & (teacher.ext_conf >= CONF)]
    for f in set(labeled_frames) | set(pseudo_frames):
        img = base / 'images' / f'{f}.jpg'
        if not img.exists():
            img.symlink_to((D / f'{f}.jpg').resolve())
        if f in set(labeled_frames):
            r = h[h.frame_id == f]
            rows = zip(r.class_name, r.cx, r.cy, r.w, r.h)
        else:
            r = t[t.frame_id == f]
            rows = zip(r.ext_class, (r.x1 + r.x2) / 2560, (r.y1 + r.y2) / 1440, (r.x2 - r.x1) / 1280, (r.y2 - r.y1) / 720)
        write_label(base / 'labels' / f'{f}.txt', rows)
    cfg = base / 'data.yaml'
    cfg.write_text(f'path: {base}\ntrain: images\nval: images\nnames: {dict(enumerate(CLASSES))}\n')
    return cfg


def main(mode):
    tr_ctx, te_ctx = pd.read_csv(D / 'train_context.csv'), pd.read_csv(D / 'test_context.csv')
    if mode == 'check':
        fold, teacher = 'd26pl', pd.read_csv(ROOT / 'data' / 'ftS_d26.csv')
        labeled = tr_ctx.frame_id[tr_ctx.frame_id.str[-15:-7] == '20261026']
        stream = pd.read_csv(D / 'train_stream_predictions.csv').frame_id.unique()
        pseudo = [f for f in stream if f[-15:-7] == '20261025']
        targets = tr_ctx.frame_id[tr_ctx.frame_id.str[-15:-7] == '20261025']
    else:
        fold, teacher = 'allpl', pd.read_csv(ROOT / 'data' / 'ftS_all.csv')
        labeled = tr_ctx.frame_id
        pseudo = list(pd.read_csv(D / 'test_stream_predictions.csv').frame_id.unique())
        targets = te_ctx.frame_id
    out = ROOT / 'data' / f'ft_{fold}.csv'
    if out.exists():
        print('skip', fold)
        return
    cfg = build(fold, list(labeled), pseudo, teacher)
    model = YOLO(ROOT / 'weights' / 'yolo11n.pt')
    model.train(data=str(cfg), epochs=EPOCHS, imgsz=IMGSZ, batch=int(os.environ.get('BATCH', 4)), workers=2, device='cpu', cache=False,
                project=str(WORK / 'runs'), name=fold, exist_ok=True, val=False, plots=False, verbose=False,
                fliplr=0.5, mosaic=1.0, close_mosaic=10, patience=0)
    predict(model, list(targets), out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'check')
