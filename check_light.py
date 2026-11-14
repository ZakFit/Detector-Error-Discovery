"""Перепроверка двух лёгких идей агента-генератора на стенде compare_ideas (общий AP, день → другой день).
Параметры зафиксированы заранее (не подбираются):
  A. Руки: «анатомический» бокс кисти из позы — центр = запястье + 0.2·(запястье − локоть), сторона = 0.8·|предплечье|,
     точки с conf ≥ 0.3. Порядок рук = ранг базы + 0.25·ранг(1 − IoU с этим боксом).
  B. Люди: OWLv2 на полном кадре видит в боксе 'a jacket'/'a bag' (IoU ≥ 0.5, score > 0.1) → +0.1 к рангу (доля) person.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from compare_ideas import DATA, FOLD, ft_file, reorder
from features import BOX, iou_matrix, load
from final_ft import ft_agree
from reannotate import agreement, amodal_iou, score

rk = lambda x: pd.Series(x).rank(pct=True).values


def hand_boxes(pose):
    out = {}
    for f, g in pose.groupby('frame_id'):
        b = []
        for e, w in ((7, 9), (8, 10)):
            ok = (g[f'k{e}_c'] >= .3) & (g[f'k{w}_c'] >= .3)
            E, W = g.loc[ok, [f'k{e}_x', f'k{e}_y']].values, g.loc[ok, [f'k{w}_x', f'k{w}_y']].values
            c, half = W + .2 * (W - E), .4 * np.linalg.norm(W - E, axis=1, keepdims=True)
            b.append(np.hstack([c - half, c + half]))
        out[f] = np.vstack(b)
    return out


def max_iou(q, boxes_by_frame, mask):
    v = np.zeros(len(q))
    for i in np.flatnonzero(mask):
        b = boxes_by_frame.get(q.frame_id[i])
        if b is not None and len(b):
            v[i] = iou_matrix(q.loc[[i], BOX].values.astype(float), b)[0].max()
    return v


def main():
    tr, *_ = load('train')
    tr = tr[tr.class_name != 'table'].reset_index(drop=True)
    tr['day'] = tr.frame_id.str[-15:-7]
    tr['agree'], tr['amodal'] = agreement(tr), amodal_iou(tr)
    hb = hand_boxes(pd.read_csv(DATA / 'second_opinion_yolo11m-pose.csv'))
    owl = pd.read_csv(DATA / 'second_opinion_owlv2.csv')
    owl = owl[owl.ext_class.isin(['a jacket', 'a bag']) & (owl.ext_conf > .1)]
    jb = {f: g[BOX].values for f, g in owl.groupby('frame_id')}
    rus = pd.read_csv(DATA / 'ruslan_oof.csv', dtype={'id': str}).set_index('id').detector_pose_lr
    rows = {}
    for day, fold in FOLD.items():
        q = tr[tr.day == day].reset_index(drop=True)
        cls = q.class_name.values
        base = (score(q) + 1 - ft_agree(q, ft_file(fold))) / 2
        H, P = cls == 'nude_hands', cls == 'person'
        pose_iou = max_iou(q, hb, H)
        flag = (max_iou(q, jb, P) >= .5).astype(float)
        r = np.nan_to_num(q.id.map(rus).values.astype(float), nan=.5)
        rb = reorder(base, r, cls, ['person'])
        v = {'база 0.842': base,
             'A руки + поза-бокс': reorder(base, rk(base) + .25 * rk(1 - pose_iou), cls, ['nude_hands']),
             'B люди + куртка/сумка': reorder(base, rk(base) + .1 * flag, cls, ['person']),
             'Ruslan': rb,
             'Ruslan + B': reorder(rb, rk(r) + .1 * flag, cls, ['person'])}
        v['Ruslan + A + B'] = reorder(v['Ruslan + B'], rk(base) + .25 * rk(1 - pose_iou), cls, ['nude_hands'])
        for k, s in v.items():
            rows.setdefault(k, {})[f'{day} все'] = average_precision_score(q.error, s)
            for c, m in (('руки', H), ('люди', P)):
                rows[k][f'{day} {c}'] = average_precision_score(q.error[m], s[m])
        print(day, 'поза-боксов найдено для рук:', int((pose_iou > 0).sum()), '/', H.sum(),
              '| куртка/сумка у людей:', int(flag.sum()), '/', P.sum(), 'ошибок среди них', int(q.error[flag > 0].sum()))
    t = pd.DataFrame(rows).T
    print(t.round(4).to_string())


if __name__ == '__main__':
    main()
