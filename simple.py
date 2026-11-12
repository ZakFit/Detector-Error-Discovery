"""Простые сабмиты: 1-confidence + максимум один понятный сигнал."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import D, build

warnings.filterwarnings('ignore')
OUT = Path(__file__).parent / 'submissions'


def class_prior(df, train):
    """Сначала класс по доле ошибок в train, внутри класса — 1-confidence."""
    rate = train.groupby('class_name').error.mean()
    return (df.class_name.map(rate).fillna(train.error.mean()) + (1 - df.confidence)) / 2


def duplicate(df, train):
    """1-confidence + штраф, если бокс накрыт более уверенным боксом того же класса (дубль/неточный бокс)."""
    return ((1 - df.confidence) + 0.5 * (df.iou_stronger_same > 0.3)) / 1.5


def temporal(df, train):
    """1 - средняя уверенность по самому кадру и соседним (±30 с), где объект найден."""
    nb = df.nb_conf_mean.fillna(df.confidence)
    return 1 - (df.confidence + nb) / 2


SIMPLE = {'class_prior': class_prior, 'duplicate': duplicate, 'temporal': temporal}


def main():
    train = build('train').reset_index(drop=True)
    test = build('test').reset_index(drop=True)
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    print('baseline', {d: round(average_precision_score(p.error, 1 - p.confidence), 4) for d, p in train.groupby('day')})
    for name, f in SIMPLE.items():
        # class_prior на дне d считается по другому дню, чтобы оценка была честной
        ap = {}
        for d, p in train.groupby('day'):
            ap[d] = round(average_precision_score(p.error, f(p, train[train.day != d])), 4)
        print(f'{name:12s}', ap)
        score = f(test, train).clip(0, 1)
        sub = template[['id']].merge(test[['id']].assign(score=score.values), on='id', validate='one_to_one')
        assert sub.score.notna().all()
        sub.to_csv(OUT / f'sub_simple_{name}.csv', index=False)


if __name__ == '__main__':
    main()
