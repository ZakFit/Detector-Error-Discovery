# Detector-Error-Discovery

Решение кейса RASCAR «Аудит детектора объектов» (Kaggle `rascar-hack-case-1`, данные — КИРА).
Камера в обеденном зале кафе каждые 30 секунд отправляет кадр, детектор ставит боксы: люди, руки, чашки, подносы.
Нужно ранжировать детекции так, чтобы ошибочные оказались в начале очереди проверки. Метрика — Average Precision.

**Итог: Public 0.91688.**

```bash
python make_submission.py        # → submission.csv (побайтно равен отправленному файлу, проверяется по sha256)
```

## Главная идея: переразметить кадр заново и сравнить

Детекция верна, если её можно сопоставить один-к-одному с ручным боксом того же класса при IoU ≥ 0.6.
Значит, лучший признак ошибки — не уверенность детектора, а то, **насколько бокс совпадает с тем, как разметил бы
человек**. Каждый кадр «переразмечается» независимыми моделями, детекция оценивается по согласию с ними:

```
кадр ─► руки / чашки / подносы:
        OWLv2 на увеличенном окне + YOLO, дообученные на ручной разметке train (полный кадр и кропы)
        → 1 − IoU, надёжность кропа, вероятность и дубли                                     (0.842 → 0.889)
     ─► люди (ручной бокс — всё тело, включая ноги под столом):
        рефайнер боксов на признаках DINOv2-L: предсказывает, где был бы ручной бокс        (0.889 → 0.911)
        + DEIMv2-M + RTMPose-m (26 точек тела) + логистическая регрессия; ранги смешаны 50/50 (0.911 → 0.91688)
```

Изменения внутри класса делаются переупорядочиванием: значения score класса сохраняются, поэтому
межклассовый порядок, проверенный на Public, не ломается.

## Путь к результату (Public)

| шаг | Public | что изменилось | код |
|---|---|---|---|
| бейзлайн `1 − confidence` | 0.641 | — | |
| переразметка открытыми моделями (OWLv2, YOLO seg/pose) | 0.712 | IoU с независимыми моделями вместо confidence | `reannotate.py` |
| + YOLO11n, дообученный на ручной разметке train | 0.842 | модель учит стиль разметчика | `final_ft.py` |
| руки/чашки/подносы: YOLO11s на полном кадре и кропах | 0.866 | надёжность по кропу | `audit_gpu_yolo.py`, `audit_gpu_eval.py` |
| + вероятности и дубли | 0.889 | | `audit_uncertainty.py`, `audit_scene_matching.py` |
| люди: геометрия по DINOv2-L, 140 px | 0.907 | | `audit_adapt_dino.py`, `audit_box_refinement.py` |
| люди: рефайнер DINOv2-L, 280 px | 0.911 | | `audit_highres_refiner.py`, `audit_export_highres.py` |
| **люди: + DEIMv2 + RTMPose + LR, ранги 50/50** | **0.91688** | | `person_pipeline/`, `final_people.py`, `make_submission.py` |

Журналы экспериментов: [NOTES.md](NOTES.md) (этапы до 0.842 и проверки идей), [EXPERIMENTS.md](EXPERIMENTS.md)
и `experiments/results/` (этапы 0.866–0.911), `person_pipeline/README.md` (модуль людей).

## Запуск

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scipy scikit-learn pillow torch torchvision timm ultralytics transformers opencv-python kagglehub
# данные (нужен Kaggle-токен и принятые правила соревнования)
KAGGLEHUB_CACHE=$PWD/data/.kagglehub .venv/bin/python -c "import kagglehub; kagglehub.competition_download('rascar-hack-case-1')"
ln -sfn .kagglehub/competitions/rascar-hack-case-1 data/raw

.venv/bin/python make_submission.py
```

`make_submission.py` собирает итог из закоммиченных выходов этапов: `submissions/team_person_refiner_280.csv`
(этап 0.911) и `person_pipeline/outputs/person_scores.csv` (модуль людей). CPU, секунды.

## Полный пересчёт

Этапы зависят друг от друга по порядку; каждый пишет CSV, который читает следующий.

```bash
# 0.842 — CPU, ~1.5 ч
.venv/bin/python second_opinion.py <yolov8s-seg.pt> <yolo11m-pose.pt>
.venv/bin/python owl_opinion.py && .venv/bin/python owl_zoom.py
.venv/bin/python finetune_yolo.py && .venv/bin/python final_ft.py ft_all.csv

# 0.866 → 0.911 — GPU (DINOv2-L, YOLO11s); промежуточные файлы пишутся в data/gpu_runs/
python audit_gpu_yolo.py ...          # YOLO11s на полном кадре и кропах, по дням и на всём train
python audit_gpu_eval.py --export crop_reliability
python audit_uncertainty.py && python audit_scene_matching.py        # → submission_multiview_duplicates.csv
python audit_adapt_dino.py && python audit_box_refinement.py         # → submission_refinement_person_geometry_half.csv
python audit_highres_refiner.py --test coarse && python audit_export_highres.py   # → submission_person_refiner_280.csv

# модуль людей — CPU, ONNX (подробно: person_pipeline/README.md)
python person_pipeline/setup_assets.py
python person_pipeline/pipeline.py extract --threads 4
python person_pipeline/pipeline.py evaluate          # проверка по дням + обучение оценщика
python person_pipeline/pipeline.py predict           # → person_pipeline/outputs/person_scores.csv

python make_submission.py
```

Скрипты `audit_*` пишут сабмиты в каталог `outputs/` уровнем выше репозитория (структура рабочей машины автора);
готовые выходы каждого этапа лежат в `submissions/` (`team_crop_person_0866.csv`, `team_multiview_duplicates.csv`,
`team_refinement_person_geometry_half.csv`, `team_person_refiner_280.csv`).

## Валидация

Train — два дня (25.10 и 26.10). Обучаемся на одном дне, проверяем на другом, класс `table` исключаем, как в test.
Изменение принимаем, только если оно даёт плюс на обоих днях. Смешивание людей 50/50, например, на отложенных днях:
0.8983 → 0.9007 и 0.8513 → 0.8652 (`repro_refiner.py`, `data/repro_refiner.csv`); на Public 0.911 → 0.91688.

## Модели и веса

| модель | для чего | источник |
|---|---|---|
| YOLO11n, YOLO11s (дообучение) | переразметка в стиле разметчика | https://github.com/ultralytics/assets/releases |
| YOLOv8s-seg, YOLO11m-pose | люди, ключевые точки тела | https://github.com/ultralytics/assets/releases |
| OWLv2 base | руки, чашки, подносы, люди по тексту | https://huggingface.co/google/owlv2-base-patch16-ensemble |
| DINOv2-L (`vit_large_patch14_dinov2.lvd142m`, timm) | признаки для рефайнера боксов людей | https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m |
| DEIMv2-M (DINOv3, COCO) | независимый детектор людей | https://huggingface.co/Intellindust/DEIMv2_DINOv3_M_COCO , код https://github.com/Intellindust-AI-Lab/DEIMv2 |
| RTMPose-m Halpe26 | 26 точек тела | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip |

Точные ревизии и sha256 весов модуля людей — `person_pipeline/assets/sources.json`.

## Структура

| файл / папка | роль |
|---|---|
| `make_submission.py` | **итоговый сабмит** 0.91688 |
| `final_people.py` | перенос порядка людей из модуля людей в сабмит (`--blend W` — смешивание рангов) |
| `person_pipeline/` | модуль людей: DEIMv2 + RTMPose + LR |
| `audit_*.py` | этапы 0.866–0.911 и эксперименты к ним (GPU) |
| `final_ft.py`, `finetune_yolo.py`, `reannotate.py`, `second_opinion.py`, `owl_*.py` | этап 0.842 |
| `features.py` | загрузка данных, IoU |
| `compare_ideas.py`, `check_light.py`, `check_theories.py`, `repro_refiner.py` | стенды проверки идей на отложенных днях |
| `experiments/` | ранние эксперименты (запуск из корня: `python -m experiments.<имя>`) |
| `submissions/` | выходы этапов и все проверочные сабмиты |
