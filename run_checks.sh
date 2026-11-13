#!/bin/bash
# Проверочные обучения для сравнения идей (учатся на 26.10, предсказывают 25.10).
# Запуск с лимитом памяти, чтобы при нехватке убивалось только обучение, а не VS Code:
#   systemd-run --user --scope -p MemoryMax=4500M ./run_checks.sh
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -f data/ft_d26_1.csv ] || SEED=1 EPOCHS=60 $PY -W ignore finetune_yolo.py d26 > data/check_seed.log 2>&1  # идея 4
IMGSZ=1280 BATCH=2 EPOCHS=60 $PY -W ignore finetune_yolo.py d26 > data/check_1280.log 2>&1                 # идея 2
EPOCHS=40 $PY -W ignore self_train.py check > data/check_selftrain.log 2>&1                                 # идея 1
echo done > data/check_done.flag
