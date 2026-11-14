# Detector Error Discovery

Ветка содержит архив результатов. Файлы `submissions/` содержат по 362 оценки в формате `id,score`. Полный список файлов, контрольные суммы SHA-256 и известные значения Public AP приведены в `results.json`. Поле `reported_public_ap` содержит `null`, если для файла нет подтверждённого результата.

| Версия | Public AP |
| --- | ---: |
| YOLO | 0,842 |
| Crop person | 0,86558 |
| Multiview duplicates | 0,88899 |
| Person geometry half | 0,90734 |
| Person refiner 280 | 0,91063 |
| Person refiner 280 + Ruslan | 0,91688 |
| CPU review v1 | 0,93745 |
| CPU joint geometry v2 | 0,97914 |
| CPU joint geometry v3 | 0,97534 |
| v2 + Ruslan rank 50 | 0,97914 |
| `submission_v2_visual_geometry_high_upside.csv` | **0,98081** |

Для последних двух кандидатов подтверждённых оценок нет. `submission_v2_final_local_geometry.csv` отличается от версии с 0,98081 на 8 строк, `submission_v2_final_extended_geometry.csv` — на 11. У расширенной версии совпадают все 50 строк top-50 и 95 строк top-100.

В `validation/` собраны метрики экспериментов и оценки на отложенных днях train. Контроль рефайнера 280 дал AP 0,94433 и 0,89410 на двух днях. Дубликаты дали 0,94739 и 0,89472. Усреднение рисков, пять геометрических признаков и нижний контекст снизили средний AP относительно контроля. Подробные результаты лежат в `validation/refiner_280_ablations.csv`.

Проверка сопоставления с порогом IoU 0,60 воспроизводит 670 из 670 меток train, если сначала максимизировать число пар, затем сумму IoU. Независимое сопоставление боксов расходится на одной строке, жадный выбор по confidence — на двух. Сравнение по каждой строке находится в `validation/official_matching_check.csv`.

`python verify_results.py` проверяет формат и контрольные суммы всех сабмитов и печатает сравнение рангов с версией 0,98081. Снимок этого сравнения сохранён в `submission_comparison.csv`.
