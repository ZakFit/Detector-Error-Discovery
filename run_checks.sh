#!/bin/bash
# Проверочные обучения для сравнения идей (учатся на 26.10, предсказывают 25.10). Запуск: setsid nohup ./run_checks.sh &
cd "$(dirname "$0")"
PY=.venv/bin/python
SEED=1 EPOCHS=60 $PY -W ignore finetune_yolo.py d26 > data/check_seed.log 2>&1        # идея 4: ещё один seed
IMGSZ=1280 EPOCHS=60 $PY -W ignore finetune_yolo.py d26 > data/check_1280.log 2>&1   # идея 2: разрешение 1280
EPOCHS=40 $PY -W ignore self_train.py check > data/check_selftrain.log 2>&1           # идея 1: самообучение
echo done > data/check_done.flag
