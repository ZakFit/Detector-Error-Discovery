"""Признаки детекций: мета-классификатор (MetaDetect/LMD) + соседние кадры как замена кандидатам до NMS."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('future.infer_string', False)  # numpy-бродкастинг по строковым колонкам
D =Path(__file__).parent / 'data' / 'raw'
W, H = 1280, 720
CLASSES = ['person', 'nude_hands', 'cup', 'tray']
BOX = ['x1', 'y1', 'x2', 'y2']


def iou_matrix(a, b):
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


def ioa_matrix(a, b):
    """Доля площади боксов a, лежащая внутри боксов b."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / (area_a[:, None] + 1e-9)


def load(part):
    """part: 'train' | 'test'. Возвращает целевые детекции, stream, контекст и время кадров."""
    targets = pd.read_csv(D / f'{part}.csv', dtype={'id': str})
    stream = pd.read_csv(D / f'{part}_stream_predictions.csv', dtype={'detection_id': str})
    ctx = pd.read_csv(D / f'{part}_context.csv')
    frames = pd.read_csv(D / f'{part}_frames.csv')
    frames['ts'] = pd.to_datetime(frames.timestamp, utc=True, format='mixed')
    aliases = json.load(open(D / 'scoring_policy.json'))['aliases']
    for df in (targets, stream):
        df['class_name'] = df.class_name.replace(aliases).astype(object)
    return targets, stream, ctx, frames


def frame_features(P, S):
    """Признаки внутри кадра. P — оцениваемые детекции кадра, S — все детекции этого кадра (включая столы)."""
    pb, sb = P[BOX].values, S[BOX].values
    pc, sc = P.class_name.values, S.class_name.values
    self_mask = P.id.values[:, None] == S.detection_id.values[None, :]
    I = np.where(self_mask, 0, iou_matrix(pb, sb))
    A = np.where(self_mask, 0, ioa_matrix(pb, sb))
    same = pc[:, None] == sc[None, :]
    table = (sc == 'table')[None, :]
    f = pd.DataFrame(index=P.index)
    f['iou_same_max'] = (I * same).max(1)
    f['iou_other_max'] = (I * ~same * ~table).max(1)
    f['n_same_overlap'] = ((I > 0.1) & same).sum(1)
    f['ioa_in_other'] = (A * ~same * ~table).max(1)       # рука внутри чашки и т.п.
    f['ioa_in_table'] = (A * table).max(1)
    f['n_same_in_frame'] = same.sum(1) - 1
    f['n_in_frame'] = (~table).sum() - 1
    # уверенность соседа того же класса, который сильнее перекрывает (дубль с более высоким conf)
    stronger = same & (S.confidence.values[None, :] > P.confidence.values[:, None]) & ~self_mask
    f['iou_stronger_same'] = (I * stronger).max(1)
    f['conf_rank_cls'] = P.groupby('class_name').confidence.rank(ascending=False, pct=True)
    # для person: где нижний край относительно стола, с которым бокс пересекается
    tb = sb[sc == 'table']
    gap = np.full(len(P), np.nan)
    if len(tb):
        ov = ioa_matrix(pb, tb) > 0.02
        for i in range(len(P)):
            if ov[i].any():
                t = tb[ov[i]]
                # насколько низ бокса ниже верхнего края стола (в высотах бокса)
                gap[i] = ((pb[i, 3] - t[:, 1]) / (pb[i, 3] - pb[i, 1])).min()
    f['bottom_vs_table_top'] = gap
    return f


def temporal_features(P, nbs):
    """Соседние кадры как кандидаты: сопоставление, консенсусный бокс и его согласие с текущим."""
    pb, pc = P[BOX].values, P.class_name.values
    conf = P.confidence.values
    f = pd.DataFrame(index=P.index)
    acc_box = pb * conf[:, None]
    acc_w = conf.copy()
    ious, confs, n_avail = [], [], 0
    for nb in nbs:
        if nb is None:
            continue
        n_avail += 1
        Q = nb
        I = iou_matrix(pb, Q[BOX].values) * (pc[:, None] == Q.class_name.values[None, :])
        j = I.argmax(1) if I.shape[1] else np.zeros(len(P), int)
        best = I.max(1) if I.shape[1] else np.zeros(len(P))
        qconf = Q.confidence.values[j] if len(Q) else np.zeros(len(P))
        qconf = np.where(best > 0, qconf, 0)
        ious.append(best)
        confs.append(qconf)
        m = best > 0.3
        if len(Q):
            acc_box += np.where(m[:, None], Q[BOX].values[j] * qconf[:, None], 0)
            acc_w += np.where(m, qconf, 0)
    if ious:
        ious, confs = np.array(ious), np.array(confs)
        f['nb_iou_mean'] = ious.mean(0)
        f['nb_iou_min'] = ious.min(0)
        f['nb_iou_max'] = ious.max(0)
        f['nb_conf_mean'] = confs.mean(0)
        f['nb_hits'] = (ious > 0.5).sum(0) / len(ious)
    else:
        for c in ['nb_iou_mean', 'nb_iou_min', 'nb_iou_max', 'nb_conf_mean', 'nb_hits']:
            f[c] = np.nan
    f['n_nb'] = n_avail
    cons = acc_box / acc_w[:, None]
    f['iou_consensus'] = np.diag(iou_matrix(pb, cons))
    hgt = (pb[:, 3] - pb[:, 1])
    wdt = (pb[:, 2] - pb[:, 0])
    f['cons_dy2'] = (pb[:, 3] - cons[:, 3]) / hgt   # отрицательный — низ обрезан относительно соседей
    f['cons_dy1'] = (pb[:, 1] - cons[:, 1]) / hgt
    f['cons_dw'] = ((cons[:, 2] - cons[:, 0]) - wdt) / wdt
    f['conf_vs_nb'] = conf - (np.nanmean(np.where(confs > 0, confs, np.nan), 0) if len(confs) else np.nan)
    return f


def build(part):
    targets, stream, ctx, frames = load(part)
    targets = targets[targets.class_name.isin(CLASSES)].copy()
    S = {k: g for k, g in stream.groupby('frame_id')}
    ts = frames.set_index('frame_id').ts
    out = []
    for _, r in ctx.iterrows():
        P = targets[targets.frame_id == r.frame_id]
        if P.empty:
            continue
        nbs = [S.get(r.previous) if isinstance(r.previous, str) else None,
               S.get(r.next) if isinstance(r.next, str) else None]
        f = pd.concat([frame_features(P, S[r.frame_id]), temporal_features(P, nbs)], axis=1)
        dts = [abs((ts[n] - ts[r.frame_id]).total_seconds()) for n in (r.previous, r.next) if isinstance(n, str)]
        f['nb_dt_max'] = max(dts) if dts else np.nan
        out.append(pd.concat([P, f], axis=1))
    df = pd.concat(out)
    df['ts'] = df.frame_id.map(ts)
    df['day'] = df.ts.dt.date.astype(str)
    df['hour'] = df.ts.dt.hour + df.ts.dt.minute / 60
    w, h = df.x2 - df.x1, df.y2 - df.y1
    df['cxn'], df['cyn'] = (df.x1 + df.x2) / 2 / W, (df.y1 + df.y2) / 2 / H
    df['wn'], df['hn'] = w / W, h / H
    df['y2n'] = df.y2 / H
    df['log_area'] = np.log(w * h)
    df['aspect'] = np.log(h / w)
    df['border'] = np.minimum.reduce([df.x1, df.y1, W - df.x2, H - df.y2]) / H
    for c in CLASSES:
        df[f'is_{c}'] = (df.class_name == c).astype(int)
    return df


FEATURES = [
    'confidence', 'conf_rank_cls', 'cxn', 'cyn', 'wn', 'hn', 'y2n', 'log_area', 'aspect', 'border', 'hour',
    'iou_same_max', 'iou_other_max', 'n_same_overlap', 'ioa_in_other', 'ioa_in_table', 'n_same_in_frame',
    'n_in_frame', 'iou_stronger_same', 'bottom_vs_table_top',
    'nb_iou_mean', 'nb_iou_min', 'nb_iou_max', 'nb_conf_mean', 'nb_hits', 'n_nb', 'iou_consensus',
    'cons_dy2', 'cons_dy1', 'cons_dw', 'conf_vs_nb', 'nb_dt_max',
] + [f'is_{c}' for c in CLASSES]
