"""Проверка теорий-флагов для person (параметры фиксированы заранее, не подбираются).
T1 наименее уверенный человек в кадре (людей >= 2, conf < max conf людей кадра), w=0.05.
T2 нетипичная форма: r = ar / медиана ar person train, r > 1.6 | r < 0.62, w=0.05.
T3 касается края кадра 1280x720 (отступ 4 px), w=0.03.
T4 T2 | T3 | area < 0.5 * медиана area person train, w=0.05.
T5 T1 + T4 (флаги складываются), w=0.05.
Основы: база 0.842, Ruslan, повтор рефайнера 0.911 (если есть data/repro_refiner_oof.csv).
Применение: reorder(B, rk(B) + w*flag, cls, ['person']).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from compare_ideas import DATA, FOLD, ft_file, reorder
from features import load
from final_ft import ft_agree
from reannotate import agreement, amodal_iou, score

rk = lambda x: pd.Series(x).rank(pct=True).values
W = {'T1': .05, 'T2': .05, 'T3': .03, 'T4': .05, 'T5': .05}


def flags(q, stream, med_ar, med_area):
    ar = (q.y2 - q.y1) / (q.x2 - q.x1)
    area = (q.y2 - q.y1) * (q.x2 - q.x1)
    r = ar / med_ar
    sp = stream[stream.class_name == 'person'].groupby('frame_id').confidence.agg(['size', 'max'])
    n = q.frame_id.map(sp['size']).fillna(0).values
    mx = q.frame_id.map(sp['max']).fillna(-1).values
    P = (q.class_name == 'person').values
    t1 = (n >= 2) & (q.confidence.values < mx)
    t2 = ((r > 1.6) | (r < .62)).values
    t3 = ((q.x1 < 4) | (q.x2 > 1276) | (q.y1 < 4) | (q.y2 > 716)).values
    t4 = t2 | t3 | (area < .5 * med_area).values
    f = {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4}
    f = {k: (v & P).astype(float) for k, v in f.items()}
    f['T5'] = f['T1'] + f['T4']
    return f


def main():
    tr, st, *_ = load('train')
    tr = tr[tr.class_name != 'table'].reset_index(drop=True)
    tr['day'] = tr.frame_id.str[-15:-7]
    tr['agree'], tr['amodal'] = agreement(tr), amodal_iou(tr)
    p = tr[tr.class_name == 'person']
    med_ar = ((p.y2 - p.y1) / (p.x2 - p.x1)).median()
    med_area = ((p.y2 - p.y1) * (p.x2 - p.x1)).median()
    print(f'медиана ar person {med_ar:.3f}, медиана area {med_area:.0f}')
    rus = pd.read_csv(DATA / 'ruslan_oof.csv', dtype={'id': str}).set_index('id').detector_pose_lr
    rp = DATA / 'repro_refiner_oof.csv'
    ref = pd.read_csv(rp, dtype={'id': str}).set_index('id') if rp.exists() else None
    rows, cnt = {}, {}
    for day, fold in FOLD.items():
        q = tr[tr.day == day].reset_index(drop=True)
        cls = q.class_name.values
        P = cls == 'person'
        base = (score(q) + 1 - ft_agree(q, ft_file(fold))) / 2
        r = np.nan_to_num(q.id.map(rus).values.astype(float), nan=.5)
        bases = {'а база': base, 'б Ruslan': reorder(base, r, cls, ['person'])}
        if ref is not None and q.id[P].isin(ref.index).all():
            g1 = q.id.map(ref.g140).fillna(0).values * P
            g2 = q.id.map(ref.g280).fillna(0).values * P
            rank = lambda x: pd.Series(x).rank().values
            half = reorder(base, .5 * rank(base) + .5 * rank(g1), cls, ['person'])
            bases['в рефайнер'] = reorder(half, .5 * rank(half) + .5 * rank(g2), cls, ['person'])
        elif ref is not None:
            print(day, 'repro_refiner_oof.csv не покрывает всех person этого дня — основа (в) пропущена')
        fl = flags(q, st, med_ar, med_area)
        for k, v in fl.items():
            cnt.setdefault(k, {})[f'{day} флагов'] = int((v > 0).sum())
            cnt[k][f'{day} ошибок среди'] = int(q.error[v > 0].sum())
        cnt.setdefault('всего person', {})[f'{day} флагов'] = int(P.sum())
        cnt['всего person'][f'{day} ошибок среди'] = int(q.error[P].sum())
        for bn, B in bases.items():
            v = {(bn, '—'): B}
            for t, f in fl.items():
                v[(bn, t)] = reorder(B, rk(B) + W[t] * f, cls, ['person'])
            for k, s in v.items():
                rows.setdefault(k, {})[f'AP {day}'] = average_precision_score(q.error, s)
                rows[k][f'person {day}'] = average_precision_score(q.error[P], s[P])
    t = pd.DataFrame(rows).T
    for day in FOLD:
        t[f'Δ {day}'] = t[f'AP {day}'] - t.groupby(level=0)[f'AP {day}'].transform('first')
    t = t[[c for d in FOLD for c in (f'AP {d}', f'Δ {d}')] + [f'person {d}' for d in FOLD]]
    print(t.round(4).to_string())
    te, ts, *_ = load('test')
    te = te[te.class_name != 'table'].reset_index(drop=True)
    for k, v in flags(te, ts, med_ar, med_area).items():
        cnt[k]['test флагов'] = int((v > 0).sum())
    cnt['всего person']['test флагов'] = int((te.class_name == 'person').sum())
    print(pd.DataFrame(cnt).T.to_string())


if __name__ == '__main__':
    main()
