"""Оценка переразметки дообученным YOLO (finetune_yolo.py): IoU детекции с его боксами того же класса.
Обучение на одном дне → проверка на другом (ft_d25 предсказывает 26.10, ft_d26 — 25.10)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import BOX, iou_matrix, load

ROOT = Path(__file__).parent


def ft_iou(df, path, min_conf=0.1):
    ft = pd.read_csv(path)
    frames = {k: g for k, g in ft[ft.ext_conf >= min_conf].groupby('frame_id')}
    out = []
    for _, p in df.iterrows():
        g = frames.get(p.frame_id)
        g = g[g.ext_class == p.class_name] if g is not None else None
        out.append(float(iou_matrix(p[BOX].values.astype(float)[None], g[BOX].values)[0].max())
                   if g is not None and len(g) else 0.0)
    return np.array(out)


def main():
    from reannotate import agreement, amodal_iou, score
    tr, *_ = load('train')
    tr = tr[tr.class_name != 'table'].reset_index(drop=True)
    tr['day'] = tr.frame_id.str[-15:-7]
    tr['agree'], tr['amodal'] = agreement(tr), amodal_iou(tr)
    for fold, day in (('d25', '20261026'), ('d26', '20261025')):
        path = ROOT / 'data' / f'ft_{fold}.csv'
        if not path.exists():
            continue
        q = tr[tr.day == day].copy()
        for thr in (0.05, 0.1, 0.25):
            q['ft'] = ft_iou(q, path, thr)
            auc = {c: round(roc_auc_score(x.error, -x.ft), 3) for c, x in q.groupby('class_name') if x.error.nunique() > 1}
            print(f'held-out {day} thr={thr} AUC ft {auc} | AP ft-only {average_precision_score(q.error, 1 - q.ft):.4f}')
        q['ft'] = ft_iou(q, path, 0.1)
        base = {c: round(roc_auc_score(x.error, -x.agree), 3) for c, x in q.groupby('class_name') if x.error.nunique() > 1}
        print(f'  reannotate agree AUC {base} | AP reannotate {average_precision_score(q.error, score(q)):.4f}'
              f' | AP mean(reannotate, ft) {average_precision_score(q.error, (score(q) + 1 - q.ft) / 2):.4f}')


def compare_ensemble(day='20261025', fold='d26'):
    """Итоговая формула final_ft на отложенном дне: YOLO11n, YOLO11s и их ансамбль (голый IoU и IoU × уверенность)."""
    from final_ft import ft_agree
    from reannotate import agreement, amodal_iou, score
    tr, *_ = load('train')
    q = tr[(tr.class_name != 'table') & (tr.frame_id.str[-15:-7] == day)].reset_index(drop=True)
    q['agree'], q['amodal'] = agreement(q), amodal_iou(q)
    files = {'n': ROOT / 'data' / f'ft_{fold}.csv', 's': ROOT / 'data' / f'ft_{fold}_s.csv'}
    print(f'held-out {day}: reannotate only AP {average_precision_score(q.error, score(q)):.4f}')
    for weighted in (False, True):
        for name, keys in (('n', ['n']), ('s', ['s']), ('n+s', ['n', 's'])):
            ft = np.mean([ft_agree(q, files[k], weighted=weighted) for k in keys], axis=0)
            s = (score(q) + 1 - ft) / 2
            auc = {c: round(roc_auc_score(x.error, s[x.index]), 3) for c, x in q.groupby('class_name') if x.error.nunique() > 1}
            print(f'  {"IoU×conf" if weighted else "IoU     "} {name:4s} AP {average_precision_score(q.error, s):.4f}  AUC {auc}')


if __name__ == '__main__':
    import sys
    compare_ensemble() if 'ensemble' in sys.argv else main()
