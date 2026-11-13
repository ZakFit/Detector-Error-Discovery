"""Сравнение 8 идей на одной схеме: модели учились на одном дне, проверка на другом (основной показатель —
25.10, модель d26; вторичный — 26.10, модель d25). Метрика — общий AP без table, как на Kaggle.
База — рецепт 0.842: score = (переразметка + 1 − IoU с дообученной YOLO) / 2.
Если для идеи нет нужного файла (обучение ещё идёт), она пропускается.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import BOX, D, iou_matrix, load
from final_ft import ft_agree
from reannotate import agreement, amodal_iou, score

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
NONP = ['nude_hands', 'cup', 'tray']
FOLD = {'20261025': 'd26', '20261026': 'd25'}


def reorder(base, key, cls, classes):
    """Внутри classes — порядок по key (больше = подозрительнее), значения score класса из base."""
    out = base.copy()
    for c in classes:
        idx = np.where(cls == c)[0]
        out[idx[np.argsort(key[idx], kind='stable')]] = np.sort(base[idx])
    return out


def ft_file(fold, suffix=''):
    p = DATA / f'ft_{fold}{suffix}.csv'
    return p if p.exists() else None


def objectlab(q, path):
    """ObjectLab (cleanlab): наши детекции — «метки», боксы YOLO — «предсказания». Возвращает подозрительность."""
    from cleanlab.object_detection.rank import compute_badloc_box_scores, compute_swap_box_scores
    classes = ['person', 'nude_hands', 'cup', 'tray']
    ft = pd.read_csv(path)
    labels, preds, rows = [], [], []
    for f, P in q.groupby('frame_id'):
        labels.append({'bboxes': P[BOX].values.astype(float), 'labels': P.class_name.map(classes.index).values})
        g = ft[ft.frame_id == f]
        preds.append(np.array([g[g.ext_class == c][BOX + ['ext_conf']].values.astype(float).reshape(-1, 5)
                               for c in classes], dtype=object))
        rows.append(P.index.values)
    bad = compute_badloc_box_scores(labels=labels, predictions=preds)
    swap = compute_swap_box_scores(labels=labels, predictions=preds)
    out = pd.Series(0.0, index=q.index)
    for r, b, s in zip(rows, bad, swap):
        out[r] = 1 - np.minimum(b, s)
    return out.values


def main():
    tr, *_ = load('train')
    tr = tr[tr.class_name != 'table'].reset_index(drop=True)
    tr['day'] = tr.frame_id.str[-15:-7]
    tr['agree'], tr['amodal'] = agreement(tr), amodal_iou(tr)
    oof = pd.read_csv(DATA.parent / 'data' / 'ruslan_oof.csv', dtype={'id': str}).set_index('id') \
        if (DATA / 'ruslan_oof.csv').exists() else None
    results = {}
    for day, fold in FOLD.items():
        q = tr[tr.day == day].reset_index(drop=True)
        cls, P = q.class_name.values, (q.class_name == 'person').values
        re = score(q)
        base = (re + 1 - ft_agree(q, ft_file(fold))) / 2
        ap = lambda s: round(average_precision_score(q.error, s), 4)
        r = {'база (рецепт 0.842)': ap(base)}

        def model_variant(name, path):
            """Улучшенный имитатор разметчика: во всех классах и только в руках/чашках/подносах."""
            if path is None:
                return
            new = (re + 1 - ft_agree(q, path)) / 2
            r[f'{name}: все классы'] = ap(new)
            r[f'{name}: люди как в базе'] = ap(reorder(base, new, cls, NONP))

        model_variant('1 самообучение', ft_file(f'{fold}pl'))
        model_variant('2 разрешение 1280', ft_file(fold, '_1280'))
        seeds = [p for p in (ft_file(fold), ft_file(fold, '_1')) if p]
        if len(seeds) > 1:
            ens = (re + 1 - np.mean([ft_agree(q, p, weighted=True) for p in seeds], axis=0)) / 2
            r['4 ансамбль seed (люди как в базе)'] = ap(reorder(base, ens, cls, NONP))
            # 3: свой подход для класса — люди база, руки/чашки сильный ансамбль, подносы простой (база)
            r['3 по классам: руки/чашки ансамбль'] = ap(reorder(base, ens, cls, ['nude_hands', 'cup']))
        if oof is not None:
            pz = q.id.map(oof.detector_pose_lr).values.astype(float)
            r['5 люди — пайплайн Ruslan'] = ap(reorder(base, np.nan_to_num(pz, nan=0.5), cls, ['person']))
        r['6 ObjectLab (люди как в базе)'] = ap(reorder(base, objectlab(q, ft_file(fold)), cls, NONP))
        flo = DATA / f'florence_{day}.csv'
        if flo.exists():
            fl = pd.read_csv(flo)
            fi = np.array([iou_matrix(p[BOX].values.astype(float)[None],
                                      fl[(fl.frame_id == p.frame_id) & (fl.ext_class == p.class_name)][BOX].values)[0].max()
                           if len(fl[(fl.frame_id == p.frame_id) & (fl.ext_class == p.class_name)]) else 0.0
                           for _, p in q.iterrows()])
            r['7 Florence-2 (люди как в базе)'] = ap(reorder(base, (re + 1 - fi) / 2 + base, cls, NONP))
        r['8 люди вниз'] = ap(np.where(P, base / 2, 0.5 + base / 2))
        results[day] = r
    table = pd.DataFrame(results)
    table['Δ 25.10'] = (table['20261025'] - table.loc['база (рецепт 0.842)', '20261025']).round(4)
    table['Δ 26.10'] = (table['20261026'] - table.loc['база (рецепт 0.842)', '20261026']).round(4)
    pd.set_option('display.width', 200)
    print(table.to_string())
    table.to_csv(DATA / 'compare_ideas.csv')


if __name__ == '__main__':
    main()
