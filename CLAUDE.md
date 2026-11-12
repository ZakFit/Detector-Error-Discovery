# CLAUDE.md

Kaggle `rascar-hack-case-1` (хакатон RASCAR, кейс КИРА): ранжировать детекции детектора кафе по вероятности ошибки.
Метрика — Average Precision по `error`. Финальный сабмит 14.11.2026 23:59 МСК, 10 сабмитов в сутки (обновляются ночью).
Полное условие — `data/raw/TASK.txt`. Журнал экспериментов и Public-скоров — **[NOTES.md](NOTES.md), читать первым**.

## Метка (ключ к задаче)
Детекция верна (`error=0`), если на кадре её можно сопоставить один-к-одному с ручным боксом того же класса при IoU ≥ 0.6.
Иначе ошибка: дубль, ложное срабатывание, не тот класс, неточная геометрия. Классы в test: person, nude_hands, cup, tray.
Ручной бокс `person` охватывает всё тело, включая ноги под столом. В ручной разметке алиас `nude_hand` → `nude_hands`.

## Окружение
```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy scikit-learn scipy pillow lightgbm torch torchvision ultralytics transformers kagglehub
# данные (нужен Kaggle-токен и принятые правила соревнования):
KAGGLEHUB_CACHE=$PWD/data/.kagglehub .venv/bin/python -c "import kagglehub; kagglehub.competition_download('rascar-hack-case-1')"
ln -sfn .kagglehub/competitions/rascar-hack-case-1 data/raw
```
pandas 3: в `features.py` выставлено `future.infer_string=False`, иначе numpy-бродкастинг строковых колонок падает.

## Текущее лучшее решение — переразметка (`reannotate.py`)
Идея: confidence детектора на test переносится плохо; лучше заново «разметить» кадр независимыми открытыми моделями
и оценивать детекцию по IoU с их боксами (это напрямую эмулирует правило метки).
- руки / чашки / подносы — OWLv2 на увеличенном окне вокруг детекции (`owl_zoom.py`);
- люди — среднее: IoU с OWLv2 + yolov8s-seg + yolo11m-pose, IoU с «полным» боксом тела по ключевым точкам pose, confidence;
- `score = 1 − согласие` (+0.01·(1−conf) для ничьих). Обучения нет.

Запуск: `.venv/bin/python reannotate.py` → `submissions/sub_reannotate_person3.csv` (+ AP по дням train в консоль).
Ответы моделей уже посчитаны и лежат в `data/*.csv` (в git). Пересчёт с нуля, CPU ~1 ч:
`second_opinion.py <yolov8s-seg.pt> <yolo11m-pose.pt>` → `owl_opinion.py` → `owl_zoom.py`.

## Веса (для отчёта финалистов — указывать название и URL)
- YOLOv8s-seg, YOLO11m-pose — https://github.com/ultralytics/assets/releases (официальные COCO-веса ultralytics)
- OWLv2 — https://huggingface.co/google/owlv2-base-patch16-ensemble

## Файлы
| файл | что |
|---|---|
| `features.py` | загрузка данных, IoU, табличные признаки (геометрия, перекрытия, соседние кадры ±30 с) |
| `second_opinion.py`, `owl_opinion.py`, `owl_zoom.py` | прогон независимых моделей → `data/*.csv` |
| `opinion_features.py` | признаки согласия с YOLO seg/pose (запястья и т.п.) |
| `reannotate.py` | **лучшее решение** |
| `final.py` | предыдущее: 1−(conf+OWL)/2 + перестановка внутри классов по зуму/коже/запястью (0.684) |
| `solve.py`, `simple.py`, `rerank.py`, `image_features.py`, `eda.py` | ранние эксперименты, все хуже бейзлайна на Public |

## Валидация и уроки (подробно — NOTES.md)
- Локально: AP отдельно по дням train (25.10 и 26.10), класс table исключён. Train всего 198 строк — легко переобучиться.
- Улучшение засчитываем, только если оно есть **на обоих днях**; признаки, отобранные перебором, на Public проваливались.
- Train → Public систематически теряет ~0.2 (бейзлайн 0.79 → 0.64). Модель Public по скорам: доли ошибок по классам
  как в train (person ~30%, остальные ~60%) — узкое место в порядке ВНУТРИ класса, особенно person.
- Не сработало: табличные признаки/LR/бустинг, фон пустого зала, порядок по классам, калибровка по классам.
- Камера в test сдвинута ~30 px относительно train — признаки абсолютного положения не переносятся.

## Правила работы
- Ручная разметка test и подстановка ответов запрещены правилами (дисквалификация). Смотреть кадры для поиска
  правил можно — правило потом одинаково применяется кодом ко всем строкам.
- Сабмиты ограничены: отправлять только варианты с приростом на обоих днях train или осознанные проверки гипотез;
  после каждого результата обновлять NOTES.md.
- Вычисления на ноутбуке без GPU: долгие прогоны делать возобновляемыми и с частичным сохранением.
