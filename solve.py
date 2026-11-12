"""Пайплайн: признаки -> модели -> валидация по дням и группам кадров -> ансамбль -> submission.csv."""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import D, FEATURES, build

warnings.filterwarnings('ignore')
OUT = Path(__file__).parent / 'submissions'


# Признаки отобраны жадно от confidence (критерий — AP по дням + GroupKFold по кадрам)
CORE = ['confidence', 'conf_vs_nb', 'border', 'iou_stronger_same', 'cons_dw']
EXTENDED = CORE + ['hn', 'cyn', 'is_tray']


def logreg(C=0.3):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def models():
    """Модель -> (фабрика, признаки)."""
    return {
        'lr_core': (logreg, CORE),
        'lr_ext': (logreg, EXTENDED),
        'et_core': (lambda: make_pipeline(SimpleImputer(strategy='median'),
                                          ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=5, max_features=0.6,
                                                               random_state=0, n_jobs=-1)), CORE),
        'lgbm_all': (lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5,
                                                verbose=-1, random_state=0), FEATURES),
    }


BLENDS = {
    'sub_core': ['lr_core'],
    'sub_ext': ['lr_ext'],
}


def rank01(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def blend(preds, names):
    """Средний ранг выбранных моделей."""
    return np.mean([rank01(preds[n]) for n in names], axis=0)


def add_blends(preds, idx_groups):
    """Бленды ранжируются внутри каждой группы (дня/фолда), как считается AP."""
    for b, names in BLENDS.items():
        preds[b] = np.zeros(len(next(iter(preds.values()))))
        for idx in idx_groups:
            preds[b][idx] = blend({n: preds[n][idx] for n in names}, names)
    return preds


def cv(df, splits):
    """Out-of-fold предсказания всех моделей для заданного разбиения."""
    y = df.error.values
    oof = {k: np.zeros(len(df)) for k in models()}
    for tr_idx, va_idx in splits:
        for name, (make, feats) in models().items():
            m = make().fit(df[feats].iloc[tr_idx], y[tr_idx])
            oof[name][va_idx] = m.predict_proba(df[feats].iloc[va_idx])[:, 1]
    return add_blends(oof, [va for _, va in splits])


def report(df, oof, title):
    print(f'\n--- {title}')
    scores = dict(oof, baseline=-df.confidence.values)
    rows = {}
    for name, s in scores.items():
        r = {d: average_precision_score(p.error, s[p.index]) for d, p in df.groupby('day')}
        r['all'] = average_precision_score(df.error, s)
        rows[name] = r
    print(pd.DataFrame(rows).T.round(4))


def main():
    train, test = build('train'), build('test')
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    print('train', train.shape, 'test', test.shape)

    days = sorted(train.day.unique())
    day_splits = [(np.where(train.day != d)[0], np.where(train.day == d)[0]) for d in days]
    report(train, cv(train, day_splits), 'Валидация: обучение на одном дне, проверка на другом')

    # более устойчивая оценка: GroupKFold по кадрам, несколько перестановок
    aps = {}
    for seed in range(5):
        frames = train.frame_id.unique()
        fmap = dict(zip(frames, np.random.RandomState(seed).permutation(len(frames))))
        splits = list(GroupKFold(5).split(train, groups=train.frame_id.map(fmap)))
        o = cv(train, splits)
        o['baseline'] = -train.confidence.values
        for k, v in o.items():
            aps.setdefault(k, []).append(average_precision_score(train.error, v))
    print('\n--- GroupKFold(5) по кадрам x5 seeds, AP mean ± std')
    for k, v in aps.items():
        print(f'  {k:10s} {np.mean(v):.4f} ± {np.std(v):.4f}')

    coef = models()['lr_core'][0]().fit(train[CORE], train.error)[-1].coef_[0]
    print('\n--- Коэффициенты lr_core (стандартизованные)\n', pd.Series(coef, CORE).round(3).to_string())

    # финальное обучение на всём train и сабмиты
    preds = {name: make().fit(train[feats], train.error).predict_proba(test[feats])[:, 1]
             for name, (make, feats) in models().items()}
    preds = add_blends(preds, [np.arange(len(test))])
    template = pd.read_csv(D / 'sample_submission.csv', dtype={'id': str})
    OUT.mkdir(exist_ok=True)
    for b in BLENDS:
        sub = template[['id']].merge(test[['id']].assign(score=preds[b]), on='id', how='left', validate='one_to_one')
        assert sub.score.notna().all(), 'есть id без предсказания'
        assert np.isfinite(sub.score).all() and sub.score.between(0, 1).all()
        sub.to_csv(OUT / f'{b}.csv', index=False)
        print(f'{OUT / b}.csv: {len(sub)} строк')


if __name__ == '__main__':
    main()
