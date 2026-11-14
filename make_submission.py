"""Итоговый сабмит команды — Public 0.91688. Одна команда, CPU, секунды:

    python make_submission.py            → submission.csv

Что происходит: берётся этап 0.911 (submissions/team_person_refiner_280.csv — руки/чашки/подносы и рефайнер людей)
и порядок людей смешивается 50/50 по рангам с оценками пайплайна людей (person_pipeline/outputs/person_scores.csv).
Значения score людей берутся из этапа 0.911, поэтому межклассовый порядок не меняется (см. final_people.py).
Как пересчитать этапы с нуля — README.md, раздел «Полный пересчёт».
"""
import hashlib
import shutil
import sys
from pathlib import Path

import final_people

ROOT = Path(__file__).parent
STAGE_0911 = ROOT / 'submissions' / 'team_person_refiner_280.csv'
PEOPLE = ROOT / 'person_pipeline' / 'outputs' / 'person_scores.csv'
SENT_SHA256 = 'cc46e7cd286724730c734426e7cfc7231c46abe45d6a2d26bf02ddec18361442'  # файл, отправленный на Kaggle (0.91688)


def main():
    sys.argv = ['final_people.py', str(STAGE_0911), str(PEOPLE), '--blend', '0.5']
    final_people.main()
    out = ROOT / 'submissions' / f'{STAGE_0911.stem}_people_blend0.5.csv'
    shutil.copy(out, ROOT / 'submission.csv')
    digest = hashlib.sha256((ROOT / 'submission.csv').read_bytes()).hexdigest()
    same = 'совпадает с отправленным (Public 0.91688)' if digest == SENT_SHA256 else 'НЕ совпадает с отправленным'
    print('submission.csv готов, sha256', digest, '—', same)


if __name__ == '__main__':
    main()
