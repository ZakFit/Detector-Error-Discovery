"""Признаки по изображениям и машинному пулу (идеи 3-4): фон пустого зала, движение, статичные конфузеры.

Всё детерминированно, без обучаемых моделей. Работаем в масштабе пула 640x360 (координаты / 2).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from features import BOX, D, iou_matrix, load

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'data' / 'cache'
EMPTY_HOURS = [0, 1, 2, 3, 4]


def gray(path, size=(640, 360)):
    img = Image.open(path).convert('L')
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def background():
    """Медиана кадров пустого зала из пула."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / 'background_empty.npy'
    if path.exists():
        return np.load(path)
    m = pd.read_csv(D / 'machine_frames.csv')
    m['ts'] = pd.to_datetime(m.timestamp, utc=True, format='mixed')
    sel = m[m.ts.dt.hour.isin(EMPTY_HOURS)].iloc[::4]
    bg = np.median(np.stack([gray(D / p) for p in sel.image_path]), axis=0)
    np.save(path, bg)
    return bg


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def crop(img, box):
    x1, y1, x2, y2 = (np.asarray(box) / 2).round().astype(int)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = max(x2, x1 + 2), max(y2, y1 + 2)
    return img[y1:y2, x1:x2]


def pool_presence():
    """Для каждого класса — предсказания пула, разделённые на часы пустого зала и рабочие часы."""
    m = pd.read_csv(D / 'machine_frames.csv')
    m['ts'] = pd.to_datetime(m.timestamp, utc=True, format='mixed')
    p = pd.read_csv(D / 'machine_predictions.csv')
    p = p[p.class_name != 'table'].merge(m[['frame_id', 'ts']], on='frame_id')
    p['is_empty'] = p.ts.dt.hour.isin(EMPTY_HOURS)
    busy_frames = (~m.ts.dt.hour.isin(EMPTY_HOURS)).sum()
    return p, busy_frames, m.ts.dt.hour.isin(EMPTY_HOURS).sum()


def build_image_features(part):
    targets, stream, ctx, frames = load(part)
    targets = targets[targets.class_name != 'table']
    bg = background()
    pool, n_busy, n_empty = pool_presence()
    pool_by_cls = {c: g for c, g in pool.groupby('class_name')}
    imgs = {}

    def img(fid):
        if fid not in imgs:
            imgs[fid] = gray(D / f'{fid}.jpg')
        return imgs[fid]

    rows = []
    for _, r in ctx.iterrows():
        P = targets[targets.frame_id == r.frame_id]
        if P.empty:
            continue
        cur = img(r.frame_id)
        nbs = [n for n in (r.previous, r.next) if isinstance(n, str)]
        for _, p in P.iterrows():
            box = p[BOX].values.astype(float)
            c = crop(cur, box)
            f = {'id': p.id}
            f['ncc_bg'] = ncc(c, crop(bg, box))
            f['diff_bg'] = float(np.abs(c - crop(bg, box)).mean() / (c.std() + 1))
            f['ncc_nb'] = np.mean([ncc(c, crop(img(n), box)) for n in nbs]) if nbs else np.nan
            f['motion_nb'] = np.mean([np.abs(c - crop(img(n), box)).mean() for n in nbs]) if nbs else np.nan
            f['crop_std'] = float(c.std())
            # статичные конфузеры: тот же класс на том же месте в пуле
            q = pool_by_cls.get(p.class_name)
            if q is not None and len(q):
                iou = iou_matrix(box[None], q[BOX].values)[0]
                hit = iou > 0.6
                f['pool_busy_rate'] = q[hit & ~q.is_empty.values].frame_id.nunique() / n_busy
                f['pool_empty_rate'] = q[hit & q.is_empty.values].frame_id.nunique() / n_empty
                f['pool_near_rate'] = q[(iou > 0.3) & ~q.is_empty.values].frame_id.nunique() / n_busy
            rows.append(f)
    return pd.DataFrame(rows)


IMAGE_FEATURES = ['ncc_bg', 'diff_bg', 'ncc_nb', 'motion_nb', 'crop_std',
                  'pool_busy_rate', 'pool_empty_rate', 'pool_near_rate']


def cached(part):
    path = CACHE / f'image_features_{part}.pkl'
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        build_image_features(part).to_pickle(path)
    return pd.read_pickle(path)


if __name__ == '__main__':
    for part in ('train', 'test'):
        print(part, cached(part).describe().T.round(3))
