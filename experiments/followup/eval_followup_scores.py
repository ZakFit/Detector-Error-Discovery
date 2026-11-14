"""Score-only ablations for the refiner and manual audit."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from assemble_precise_review import match, iou
from precise_review import BOX, window

ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / 'outputs'
OUT = OUTPUTS / 'followup_280'
AUDIT = OUTPUTS / 'manual_bbox_review'
DAYS = ('20261025', '20261026')
CLASSES = ('person', 'nude_hands', 'cup', 'tray')


def logits(score):
    return logit(np.clip(score, .001, .999))


def calibration_bias(fit):
    residual = fit.error - fit.score
    curvature = fit.score * (1 - fit.score)
    # One damped Newton update for class offsets, with a shared offset removed.
    offsets = (residual.groupby(fit.class_name).sum() / (curvature.groupby(fit.class_name).sum() + 8))
    return offsets - offsets.mean()


def group_pairs(det):
    rows = []
    for _, group in det.groupby(['frame_id', 'class_name']):
        overlaps = iou(group[BOX], group[BOX])
        for left, right in zip(*np.where(np.triu(overlaps >= .45, 1))):
            a, b = group.iloc[left], group.iloc[right]
            area_a = (a.x2-a.x1)*(a.y2-a.y1)
            area_b = (b.x2-b.x1)*(b.y2-b.y1)
            rows.append((a.name, b.name, a.score-b.score, a.confidence-b.confidence,
                         np.log(area_a/area_b), a.geometry-b.geometry, overlaps[left, right]))
    return pd.DataFrame(rows, columns=['a', 'b', 'risk', 'confidence', 'area', 'geometry', 'overlap'])


def group_risk(train, pred):
    train_pairs = group_pairs(train)
    pred_pairs = group_pairs(pred)
    a_error = train.error.reindex(train_pairs.a).to_numpy()
    b_error = train.error.reindex(train_pairs.b).to_numpy()
    contested = (a_error != b_error)
    features = ['risk', 'confidence', 'area', 'geometry']
    x = train_pairs.loc[contested, features].to_numpy()
    y = a_error[contested]
    score = pred.score.to_numpy(copy=True)
    if len(x) < 3 or pred_pairs.empty:
        return score, len(x)
    model = LogisticRegression(C=1, fit_intercept=False).fit(np.vstack([x, -x]), np.r_[y, 1-y])
    probability = model.predict_proba(pred_pairs[features].to_numpy())[:, 1]
    evidence = np.zeros(len(pred))
    weight = np.zeros(len(pred))
    for pair, p in zip(pred_pairs.itertuples(), probability):
        evidence[pair.a] += pair.overlap * (p - .5)
        evidence[pair.b] += pair.overlap * (.5 - p)
        weight[[pair.a, pair.b]] += pair.overlap
    return np.clip(score + .45 * evidence / np.maximum(weight, 1), 0, 1), len(x)


def load_rows(raw, split):
    score = pd.read_csv(OUT / f'control_280_{split}.csv')
    geometry = pd.read_csv(raw.parent / 'gpu_runs/highres_refiner' / ('test.csv' if split == 'test' else 'oof.csv'))
    det = pd.read_csv(raw / ('test.csv' if split == 'test' else 'train.csv'))
    return (score.merge(det[['id', 'confidence', *BOX]], on='id', validate='one_to_one')
            .merge(geometry[['id', 'coarse']].rename(columns={'coarse':'geometry'}), on='id', validate='one_to_one'))


def blind_references(samples):
    manual = pd.read_csv(AUDIT / 'blind_manual.csv')
    table = samples.merge(manual, on='review_number', validate='one_to_one').dropna(subset=['x1p'])
    refs = []
    for _, r in table.iterrows():
        x1, y1, x2, y2 = window(r)
        percentages = r[['x1p', 'y1p', 'x2p', 'y2p']].to_numpy(dtype=float) / 100
        box = percentages * ([x2-x1, y2-y1] * 2) + [x1,y1,x1,y1]
        refs.append(dict(frame_id=r.frame_id, class_name=r.class_name, quality=r.quality,
                         object_id=r.review_number, **dict(zip(BOX, box.clip([0,0,0,0], [1280,720,1280,720])))))
    refs = pd.DataFrame(refs)
    # Samples 20-22 are different views of the same tray in one frame.
    cluster = refs.object_id.isin([20, 21, 22])
    refs.loc[refs.object_id.eq(21), BOX] = refs.loc[cluster, BOX].median().to_numpy()
    return refs[~refs.object_id.isin([20, 22])].reset_index(drop=True)


def learn_style(refs, official):
    residuals = []
    for row in refs.itertuples():
        box = np.array([getattr(row, edge) for edge in BOX])
        gt = official.loc[(official.frame_id == row.frame_id) & (official.class_name == row.class_name), BOX].to_numpy()
        overlap = iou(box[None], gt)[0]
        if len(overlap) and overlap.max() >= .35:
            scale = np.tile(box[2:] - box[:2], 2)
            residuals.append((row.class_name, *np.clip((gt[overlap.argmax()] - box) / scale, -.5, .5)))
    data = pd.DataFrame(residuals, columns=['class_name', *BOX])
    offsets = {cls: rows[BOX].median().to_numpy()*len(rows)/(len(rows)+5) for cls, rows in data.groupby('class_name')}
    return offsets, len(data)


def apply_style(refs, offsets):
    fixed = refs.copy()
    for cls, edges in offsets.items():
        sel = fixed.class_name == cls
        boxes = fixed.loc[sel, BOX].to_numpy()
        fixed.loc[sel, BOX] = np.clip(boxes+np.tile(boxes[:, 2:]-boxes[:, :2], 2)*edges,
                                      [0,0,0,0], [1280,720,1280,720])
    return fixed


def main():
    raw = ROOT / 'work/Detector-Error-Discovery/data/raw'
    folds = {split: load_rows(raw, split) for split in (*DAYS, 'test')}
    fit_all = pd.concat([folds[d] for d in DAYS], ignore_index=True)
    raw_gt = pd.read_csv(raw / 'human_boxes_train.csv')
    raw_gt['class_name'] = raw_gt.class_name.replace({'nude_hand':'nude_hands'})
    for edge, center, size, image_size, side in zip(BOX, ('cx','cy')*2, ('w','h')*2, (1280,720)*2, (-1,-1,1,1)):
        raw_gt[edge] = (raw_gt[center] + side*.5*raw_gt[size])*image_size
    blind = pd.read_csv(AUDIT / 'blind_train_samples.csv')
    blind_refs = blind_references(blind)

    for split, pred in folds.items():
        fit = fit_all if split == 'test' else folds[next(d for d in DAYS if d != split)]
        score = expit(logits(pred.score) + pred.class_name.map(calibration_bias(fit)).fillna(0))
        group_score, pairs = group_risk(fit, pred)
        print(split, 'duplicate training pairs', pairs)
        for name, values in [('class_calibration', score), ('duplicate_pair', group_score)]:
            pred[['id','frame_id','class_name','error']].assign(score=values).to_csv(OUT/f'{name}_{split}.csv',index=False)
            if split != 'test':
                print(split, name, average_precision_score(pred.error, values))

        learn = blind_refs if split == 'test' else blind_refs[~blind_refs.frame_id.str.contains(split)]
        offsets, n = learn_style(learn, raw_gt)
        if split == 'test':
            manual_refs = pd.read_csv(AUDIT / 'manual_boxes.csv')
            det = pred
        else:
            manual_refs = blind_refs[blind_refs.frame_id.str.contains(split)]
            det = blind[blind.frame_id.str.contains(split)].reset_index(drop=True)
        before, before_iou, _ = match(det, manual_refs)
        after, after_iou, _ = match(det, apply_style(manual_refs, offsets))
        details = det[['id','class_name','error']].assign(before=before, after=after, iou_before=before_iou, iou_after=after_iou)
        details.to_csv(OUT/f'annotation_style_{split}.csv',index=False)
        if split == 'test':
            print(split, 'annotation style changes', (before != after).sum(), 'fit pairs', n)
        else:
            print(split, 'annotation style', (before == det.error).sum(), '->', (after == det.error).sum(), 'of', len(det), 'fit pairs', n)


if __name__ == '__main__':
    main()
