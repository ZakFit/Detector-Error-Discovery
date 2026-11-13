"""Люди — пайплайн Ruslan (ветка ruslan/person-pipeline: DEIMv2 + RTMPose + логистическая регрессия),
остальные классы — без изменений из базового сабмита.

Как в его `pipeline.py merge --mode rank`: люди упорядочиваются по его score, а значения score людей
берутся из базы — масштаб и порядок между классами сохраняются.
Проверка (compare_ideas.py, идея 5; модель учится на одном дне, проверка на другом):
AP 25.10 0.887 → 0.900, 26.10 0.837 → 0.883 — единственная из 8 идей с плюсом на обоих днях.

  python final_people.py <base.csv> [person_scores.csv]
  person_scores.csv по умолчанию — data/ruslan_person_scores.csv (outputs/person_scores.csv из его ветки).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import D

ROOT = Path(__file__).parent


def main():
    base_path = Path(sys.argv[1])
    people_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'data' / 'ruslan_person_scores.csv'
    test = pd.read_csv(D / 'test.csv', dtype={'id': str})
    base = pd.read_csv(base_path, dtype={'id': str})
    people = pd.read_csv(people_path, dtype={'id': str}).set_index('id').score
    person_ids = set(test.id[test.class_name == 'person'])
    assert set(people.index) == person_ids, 'score людей должны покрывать ровно всех person из test'
    idx = np.where(base.id.isin(person_ids))[0]
    new = base.id.values[idx]
    order = np.argsort(people.reindex(new).values, kind='stable')
    out = base.copy()
    out.loc[out.index[idx[order]], 'score'] = np.sort(base.score.values[idx], kind='stable')
    assert np.allclose(np.sort(out.score), np.sort(base.score)) and (out.score[~out.id.isin(person_ids)]
                                                                    == base.score[~base.id.isin(person_ids)]).all()
    dst = ROOT / 'submissions' / f'{base_path.stem}_people.csv'
    out.to_csv(dst, index=False)
    print(dst, '| person rows reordered:', len(idx))


if __name__ == '__main__':
    main()
