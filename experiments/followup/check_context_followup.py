"""Validate background and adjacent-frame signals with day-held-out scoring."""
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score as ap

from assemble_precise_review import BOX, iou, match

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / 'work/Detector-Error-Discovery/data/raw'
OUT = ROOT / 'outputs/followup_280'
DAYS = ('20261025', '20261026')
SMALL = ('cup', 'tray')
cv2.setNumThreads(1)


@lru_cache(maxsize=300)
def image(frame):
    rgb = cv2.imread(str(RAW / f'{frame}.jpg'))
    assert rgb is not None, frame
    return rgb


def bounds(row):
    return np.clip(np.rint([row.x1,row.y1,row.x2,row.y2]).astype(int), [0,0,0,0], [1280,720,1280,720])


def thumbnail(frame, box):
    x1,y1,x2,y2 = box
    return cv2.resize(image(frame)[y1:y2,x1:x2], (24,24), interpolation=cv2.INTER_AREA).astype(np.float32)


def difference(a, b):
    # Remove an additive illumination shift, while retaining differences in structure and colour.
    diff = a-b
    shift = np.clip(np.median(diff, axis=(0,1), keepdims=True), -12, 12)
    return np.mean(np.abs(diff-shift))


def backgrounds(held):
    train = pd.read_csv(RAW/'train_frames.csv', dtype={'day':str})
    train = train[train.day.ne(held) & train.image_exists]
    stream = pd.read_csv(RAW/'train_stream_predictions.csv')
    busy = stream.loc[stream.class_name.eq('person') & stream.confidence.ge(.35), 'frame_id']
    sample = train.loc[~train.frame_id.isin(busy)].iloc[::11].head(180)
    # Bank of prototypes from different times helps with day/night lighting.
    return tuple(cv2.resize(image(k),(320,180),interpolation=cv2.INTER_AREA) for k in sample.frame_id)


def background_match(row, bank):
    box = bounds(row)
    crop = thumbnail(row.frame_id, box)
    x1,y1,x2,y2 = np.rint(box/4).astype(int)
    if x2<=x1 or y2<=y1:
        return np.nan
    distances = sorted(difference(crop, cv2.resize(frame[y1:y2,x1:x2],(24,24)).astype(np.float32)) for frame in bank)
    return float(np.median(distances[:3]))


def temporal_reference(rows, context, stream):
    bank = {key: group for key,group in stream.groupby(['frame_id','class_name'])}
    tracks = []
    for frame, group in rows[rows.class_name.isin(SMALL)].groupby('frame_id'):
        if frame not in context.index:
            continue
        for row in group.itertuples():
            box = bounds(row)
            current = thumbnail(frame, box)
            choices = []
            for neighbour in context.loc[frame, ['previous','next']]:
                if not isinstance(neighbour, str) or not (RAW/f'{neighbour}.jpg').exists():
                    continue
                peers = bank.get((neighbour,row.class_name))
                motion = difference(current, thumbnail(neighbour, box))
                if peers is None or motion > 9:
                    continue
                matrix = iou(np.array([[row.x1,row.y1,row.x2,row.y2]]), peers[BOX].to_numpy())[0]
                eligible = (matrix >= .38) & (peers.confidence.to_numpy() >= .35)
                if eligible.any():
                    ref = peers.iloc[np.argmax(np.where(eligible, matrix, -1))]
                    choices.append((neighbour, motion, float(ref.confidence), ref[BOX].to_numpy(float)))
            if choices:
                for neighbour,motion,conf,box in choices:
                    tracks.append((row.id,frame,row.class_name,neighbour,motion,conf,*box))
    return pd.DataFrame(tracks,columns=['id','frame_id','class_name','neighbour','motion','confidence',*BOX])


def temporal_scores(rows, tracks):
    scores = rows.score.to_numpy().copy()
    for (_, cls), q in rows.groupby(['frame_id','class_name']):
        support = tracks[tracks.id.isin(q.id)]
        if support.empty or cls not in SMALL:
            continue
        counts = support.groupby('id').neighbour.nunique()
        support = support[support.id.isin(counts[counts >= 2].index)]
        refs = support.groupby('id')[BOX].median().to_numpy()
        if not len(refs):
            continue
        keep = []
        for box in refs:
            if not keep or iou(box[None],np.array(keep)).max() < .65:
                keep.append(box)
        matrix = iou(q[BOX].to_numpy(), np.asarray(keep))
        valid = matrix >= .66
        left,right = linear_sum_assignment(np.where(valid,len(q)+1+matrix,0),maximize=True)
        selected = q.index.to_numpy()[left[valid[left,right]]]
        scores[selected] *= .8
    return scores


def joint_teacher_scores(rows, split):
    """Ablate joint matching while keeping the reference views and blend fixed."""
    fold = {'20261025':'d26','20261026':'d25','test':'all'}[split]
    views = pd.read_csv(RAW.parent / 'gpu_runs' / f'uncertainty_{fold}.csv')
    views = views[views.confidence.ge(.10)].merge(rows[['id','frame_id']],on='id',validate='many_to_one')
    indexed = {key:q for key,q in views.groupby(['frame_id','class_name','view'])}
    independent = np.zeros(len(rows))
    joint = np.zeros(len(rows))
    nviews = views.view.nunique()
    for (frame,cls), group in rows.groupby(['frame_id','class_name']):
        if len(group)<2 or iou(group[BOX],group[BOX])[np.triu_indices(len(group),1)].max()<.45:
            continue
        for view in views.view.unique():
            ref = indexed.get((frame,cls,view))
            if ref is None:
                continue
            keep = []
            for box in ref.sort_values('confidence',ascending=False)[BOX].to_numpy():
                if not keep or iou(box[None],np.array(keep)).max()<.60:
                    keep.append(box)
            overlap = iou(group[BOX],np.asarray(keep))
            good = overlap>=.6
            a,b = linear_sum_assignment(np.where(good,len(group)+1+overlap,0),maximize=True)
            independent[group.index] += good.any(1)/nviews
            joint[group.index.to_numpy()[a[good[a,b]]]] += 1/nviews
    affected = independent>0
    independent_score = np.where(affected,.8*rows.score+.2*(1-independent),rows.score)
    joint_score = np.where(affected,.8*rows.score+.2*(1-joint),rows.score)
    return independent_score, joint_score


def main():
    official = pd.read_csv(RAW/'human_boxes_train.csv')
    official.class_name = official.class_name.replace({'nude_hand':'nude_hands'})
    for edge, center, size, sign, limit in zip(BOX,('cx','cy')*2,('w','h')*2,(-1,-1,1,1),(1280,720)*2):
        official[edge] = (official[center]+sign*official[size]/2)*limit
    official['object_id'] = official.index.astype(str)
    tables = []
    for split in (*DAYS,'test'):
        is_test = split=='test'
        part = 'test' if is_test else 'train'
        base = pd.read_csv(OUT/f'control_280_{split}.csv')
        det = pd.read_csv(RAW/f'{part}.csv')
        rows = base.merge(det[['id',*BOX]],on='id',validate='one_to_one')
        assert len(rows)==len(base) and rows.id.is_unique
        if not is_test:
            assert rows.frame_id.str[-15:-7].eq(split).all()
            assert np.array_equal(match(rows, official)[0], rows.error.to_numpy())
            print(split, 'official matching', len(rows), '/', len(rows), 'AP',ap(rows.error, rows.score),flush=True)
        ctx = pd.read_csv(RAW/f'{part}_context.csv').set_index('frame_id')
        stream = pd.read_csv(RAW/f'{part}_stream_predictions.csv')
        tracks = temporal_reference(rows,ctx,stream)
        tracks.to_csv(OUT/f'context_tracks_{split}.csv',index=False)
        bank = backgrounds(split)
        dist = np.array([background_match(r,bank) for r in rows.itertuples()])
        background = np.where(dist <= 5, rows.score+.20*(1-rows.score), rows.score)
        independent, joint = joint_teacher_scores(rows, split)
        background_relaxed = np.where(dist <= 24, rows.score+.20*(1-rows.score), rows.score)
        table = rows[['id','frame_id','class_name','error','score']].assign(split=split,temporal=temporal_scores(rows,tracks),background=background,background_relaxed=background_relaxed,background_distance=dist,independent=independent,joint=joint)
        print(split, 'temporal pairs',len(tracks),'background bank',len(bank),flush=True)
        for name in ('temporal','background','background_relaxed','independent','joint'):
            changed = table[table[name].ne(table.score)]
            print(split,name,len(changed),'true errors',None if is_test else changed.error.sum(),'AP',None if is_test else ap(table.error,table[name]),flush=True)
        tables.append(table)
    result = pd.concat(tables,ignore_index=True)
    assert not result.id.duplicated().any()
    result.to_csv(OUT/'context_validation.csv',index=False)
    oof = result[result.split.ne('test')]
    print('Pooled OOF AP', {n: ap(oof.error,oof[n]) for n in ('score','temporal','background')})


if __name__ == '__main__':
    main()
