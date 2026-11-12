"""Разведочный анализ кейса RASCAR: аудит ошибок детектора."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score

D = Path(__file__).parent / 'data' / 'raw'
W, H = 1280, 720
pd.set_option('display.width', 200)
pd.set_option('future.infer_string', False)
pd.set_option('display.max_columns', 30)


def iou_matrix(a, b):
    """IoU между наборами боксов xyxy: a (n,4), b (m,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def section(title):
    print(f'\n{"=" * 8} {title} {"=" * 8}')


train = pd.read_csv(D / 'train.csv', dtype={'id': str})
test = pd.read_csv(D / 'test.csv', dtype={'id': str})
human = pd.read_csv(D / 'human_boxes_train.csv')
tr_frames = pd.read_csv(D / 'train_frames.csv')
te_frames = pd.read_csv(D / 'test_frames.csv')
tr_ctx = pd.read_csv(D / 'train_context.csv')
te_ctx = pd.read_csv(D / 'test_context.csv')
tr_stream = pd.read_csv(D / 'train_stream_predictions.csv', dtype={'detection_id': str})
te_stream = pd.read_csv(D / 'test_stream_predictions.csv', dtype={'detection_id': str})
m_frames = pd.read_csv(D / 'machine_frames.csv')
m_pred = pd.read_csv(D / 'machine_predictions.csv')

for df in (tr_frames, te_frames, m_frames):
    df['ts'] = pd.to_datetime(df.timestamp, utc=True, format='mixed')
train = train.merge(tr_frames[['frame_id', 'ts']], on='frame_id')
test = test.merge(te_frames[['frame_id', 'ts']], on='frame_id')
for df in (train, test):
    df['day'] = df.ts.dt.date
    df['hour'] = df.ts.dt.hour
    df['w'] = df.x2 - df.x1
    df['h'] = df.y2 - df.y1
human[['x1', 'y1', 'x2', 'y2']] = np.c_[
    (human.cx - human.w / 2) * W, (human.cy - human.h / 2) * H,
    (human.cx + human.w / 2) * W, (human.cy + human.h / 2) * H]
policy = json.load(open(D / 'scoring_policy.json'))
human['class_name'] = human.class_name.replace(policy['aliases'])

section('Размеры')
print('train строк', len(train), 'кадров', train.frame_id.nunique(), '| test строк', len(test), 'кадров', test.frame_id.nunique())
print('ручных боксов', len(human), '| source:', human.source.value_counts().to_dict())
print('stream: train', len(tr_stream), 'кадров', tr_stream.frame_id.nunique(), '| test', len(te_stream), 'кадров', te_stream.frame_id.nunique())
print('пул: кадров', len(m_frames), 'предсказаний', len(m_pred))
print('пересечение кадров пула с train-целевыми:', len(set(m_frames.frame_id) & set(train.frame_id)),
      'с train stream:', len(set(m_frames.frame_id) & set(tr_stream.frame_id)))
print('stream содержит целевые детекции с теми же id: train',
      train.id.isin(tr_stream.detection_id).mean(), 'test', test.id.isin(te_stream.detection_id).mean())

section('Классы: train (без table) vs test')
tr = train[train.class_name != 'table']
cls = tr.groupby('class_name').agg(n=('error', 'size'), err=('error', 'sum'), rate=('error', 'mean'))
cls['test_n'] = test.class_name.value_counts()
print(cls.sort_values('n', ascending=False).round(3))
print('доля ошибок train без table:', round(tr.error.mean(), 3))

section('По дням')
print(tr.groupby('day').agg(frames=('frame_id', 'nunique'), n=('error', 'size'), rate=('error', 'mean')).round(3))
print(test.groupby('day').agg(frames=('frame_id', 'nunique'), n=('id', 'size')))
print(pd.crosstab(tr.class_name, tr.day, values=tr.error, aggfunc='mean').round(2))


def ap_by_day(df, score):
    return {str(d): round(average_precision_score(p.error, score[p.index]), 4) for d, p in df.groupby('day')}


section('Бейзлайны (AP по дням)')
print('константа        ', {str(d): round(p.error.mean(), 4) for d, p in tr.groupby('day')})
print('1-confidence     ', ap_by_day(tr, 1 - tr.confidence))
# априорная доля ошибок класса, посчитанная на другом дне
prior = {}
for d in tr.day.unique():
    rates = tr[tr.day != d].groupby('class_name').error.mean()
    m = tr.day == d
    prior.update(tr[m].class_name.map(rates).fillna(tr[~m].error.mean()).to_dict())
prior = pd.Series(prior)
print('class prior (CV) ', ap_by_day(tr, prior))
print('prior*(1-conf)   ', ap_by_day(tr, prior * (1 - tr.confidence)))
print('AP 1-conf внутри класса:')
for c, p in tr.groupby('class_name'):
    if 0 < p.error.sum() < len(p):
        print(f'  {c:18s} n={len(p):3d} rate={p.error.mean():.2f} AP={average_precision_score(p.error, 1 - p.confidence):.3f}')

section('Confidence: ошибки vs верные')
print(tr.groupby('error').confidence.describe().round(3))

# ---- типы ошибок через сопоставление с ручной разметкой ----
section('Типы ошибок (по ручным боксам)')
rows = []
for fid, P in tr.groupby('frame_id'):
    G = human[human.frame_id == fid]
    pb, gb = P[['x1', 'y1', 'x2', 'y2']].values, G[['x1', 'y1', 'x2', 'y2']].values
    I = iou_matrix(pb, gb)
    same = (P.class_name.values[:, None] == G.class_name.values[None, :])
    # одноклассовое сопоставление: какие GT уже заняты верными предсказаниями
    taken = set()
    for c in P.class_name.unique():
        pi, gi = np.where(P.class_name.values == c)[0], np.where(G.class_name.values == c)[0]
        if len(gi) == 0:
            continue
        sub = I[np.ix_(pi, gi)]
        r, k = linear_sum_assignment(-np.where(sub >= 0.6, 1 + sub, 0))
        taken |= {gi[kk] for rr, kk in zip(r, k) if sub[rr, kk] >= 0.6}
    PP = P[['x1', 'y1', 'x2', 'y2']].values
    Ipp = iou_matrix(PP, PP)
    np.fill_diagonal(Ipp, 0)
    for i, (_, p) in enumerate(P.iterrows()):
        s = I[i] * same[i]
        o = I[i] * ~same[i]
        dup = (Ipp[i] * (P.class_name.values == p.class_name)).max() if len(P) > 1 else 0
        best = s.argmax() if len(G) else -1
        if p.error == 0:
            t = 'ok'
        elif len(G) and s.max() >= 0.6 and best in taken:
            t = 'duplicate'
        elif len(G) and s.max() >= 0.3:
            t = 'geometry'
        elif len(G) and o.max() >= 0.5:
            t = 'wrong_class'
        else:
            t = 'false_positive'
        rows.append({'id': p.id, 'etype': t, 'best_iou_same': s.max() if len(G) else 0,
                     'best_iou_other': o.max() if len(G) else 0, 'dup_iou': dup})
et = pd.DataFrame(rows)
tr = tr.merge(et, on='id')
print(pd.crosstab(tr.class_name, tr.etype, margins=True))
g = tr[(tr.etype == 'geometry') & (tr.class_name == 'person')]
if len(g):
    hp = human[human.class_name == 'person']
    print('person geometry: медианный IoU', round(g.best_iou_same.median(), 3))

section('Геометрия person: ошибки vs верные')
pp = tr[tr.class_name == 'person'].copy()
pp['aspect'] = pp.h / pp.w
pp['y2n'] = pp.y2 / H
print(pp.groupby('error')[['w', 'h', 'aspect', 'y2n', 'confidence']].median().round(3))

section('Внутрикадровые дубли (тот же класс, IoU>0.5)')
tr['has_dup'] = tr.dup_iou > 0.5
print(tr.groupby('has_dup').error.agg(['size', 'mean']).round(3))

# ---- временная устойчивость ----
section('Устойчивость в соседних кадрах')


def temporal(df, stream, ctx):
    S = {f: g for f, g in stream.groupby('frame_id')}
    out = []
    for _, r in ctx.iterrows():
        P = df[df.frame_id == r.frame_id]
        if P.empty:
            continue
        b = P[['x1', 'y1', 'x2', 'y2']].values
        feats = {'n_nb': 0, 'nb_iou_sum': np.zeros(len(P))}
        for nb in (r.previous, r.next):
            if isinstance(nb, str) and nb in S:
                Q = S[nb]
                I = iou_matrix(b, Q[['x1', 'y1', 'x2', 'y2']].values)
                I = I * (P.class_name.values[:, None] == Q.class_name.values[None, :])
                feats['nb_iou_sum'] += I.max(1) if I.shape[1] else 0
                feats['n_nb'] += 1
        out.append(pd.DataFrame({'id': P.id.values, 'nb_iou_mean': feats['nb_iou_sum'] / max(feats['n_nb'], 1),
                                 'n_nb': feats['n_nb']}))
    return pd.concat(out)


tr = tr.merge(temporal(tr, tr_stream, tr_ctx), on='id', how='left')
tr['nb_bin'] = pd.cut(tr.nb_iou_mean, [-0.01, 0.01, 0.5, 0.8, 1.0])
print(tr.groupby('nb_bin', observed=True).error.agg(['size', 'mean']).round(3))
print('AP (1 - nb_iou_mean)', ap_by_day(tr.set_index(tr.index), 1 - tr.nb_iou_mean))
print('число соседей у целевых кадров: train', tr.groupby('n_nb').size().to_dict())
te_t = temporal(test, te_stream, te_ctx)
print('                                 test', te_t.groupby('n_nb').size().to_dict())

section('Время суток')
print(tr.groupby('hour').error.agg(['size', 'mean']).round(2).T)
print('test часы:', test.hour.value_counts().sort_index().to_dict())

section('Машинный пул')
m_frames['day'] = m_frames.ts.dt.date
print(m_frames.groupby('day').size().to_dict())
dt = m_frames.sort_values('ts').ts.diff().dt.total_seconds()
print('интервалы, с:', dt.describe().round(1).to_dict())
print('классы пула:', m_pred.class_name.value_counts().to_dict())
print('классы test stream:', te_stream.class_name.value_counts().to_dict())
