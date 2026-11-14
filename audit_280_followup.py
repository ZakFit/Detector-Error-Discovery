"""Controlled follow-up experiments against the cached 280 refiner."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

import audit_highres_refiner as control
from audit_highres_refiner import BOX, CLASSES, D, WORK, enc, iou_matrix, rank_mix, supervision

OUT = Path(__file__).resolve().parents[2] / 'outputs' / 'followup_280'
SEEDS = (117, 312, 614)
DAYS = ('20261025', '20261026')


def inputs():
    train, *_ = control.load('train')
    test, *_ = control.load('test')
    truth = enc.targets()
    rows = pd.concat([train.assign(part='train'), test.assign(part='test'),
                      enc.candidates(truth).assign(part='synth')], ignore_index=True)
    return rows[rows.class_name.isin(CLASSES)].reset_index(drop=True), truth


def base_for(det, split):
    if split == 'test':
        scores = pd.read_csv(OUT.parent / 'submission_refinement_person_geometry_half.csv').set_index('id').score
    else:
        scores = pd.read_csv(enc.WORK / 'box_refinement_oof.csv').set_index('id').person_geometry_half
    base = det.id.map(scores).to_numpy()
    assert np.isfinite(base).all()
    return base


def split_indices(rows, split):
    day = rows.frame_id.str[-15:-7]
    fit = (rows.part != 'test') & ((day != split) if split != 'test' else True)
    pred = (rows.part == 'test') if split == 'test' else ((rows.part == 'train') & (day == split))
    return np.flatnonzero(fit), np.flatnonzero(pred)


def risk_parts(det, prediction):
    boxes = det[BOX].to_numpy()
    scale = np.tile(boxes[:, 2:] - boxes[:, :2], 2)
    revised = np.clip(boxes + prediction[:, :4] * scale, [0, 0, 0, 0], [1280, 720, 1280, 720])
    revised[:, 2:] = np.maximum(revised[:, 2:], revised[:, :2] + 1e-6)
    return np.diag(iou_matrix(boxes, revised)), 1 / (1 + np.exp(-prediction[:, 4].clip(-40, 40)))


def aggregation(det, split, base):
    seeds = [np.load(WORK / f'{split}_coarse_{seed}.npy') for seed in SEEDS]
    assert all(x.shape == (len(det), 6) for x in seeds)
    orig = control.geometry(det, np.mean(seeds, axis=0))
    overlaps, existence = np.array([risk_parts(det, x) for x in seeds]).transpose(1, 0, 2)
    signals = {'control_280': orig, 'mean_risk': (1 - overlaps * existence).mean(0),
               'mean_threshold_risk': (1 - (overlaps >= .6) * existence).mean(0)}
    reference = pd.read_csv(WORK / ('test.csv' if split == 'test' else 'oof.csv'))
    reference = reference if split == 'test' else reference[reference.day.astype(str) == split]
    assert reference.id.tolist() == det.id.tolist()
    np.testing.assert_allclose(orig, reference.coarse, atol=1e-10, rtol=0)
    results = {k: rank_mix(det, base, v, .5, True) for k, v in signals.items()}
    np.testing.assert_allclose(results['control_280'], reference['coarse_rank_0.5'], atol=1e-12, rtol=0)
    return results


def crop_geometry(rows):
    boxes = rows[BOX].to_numpy()
    margin = np.tile((boxes[:, 2:] - boxes[:, :2]) * .25, 2)
    start = np.maximum(np.trunc(boxes[:, :2] - margin[:, :2]), 0)
    end = np.minimum(np.ceil(boxes[:, 2:] + margin[:, 2:]), [1280, 720])
    size = end - start
    assert (size > 0).all()
    relative = (boxes - np.tile(start, 2)) / np.tile(size, 2)
    return np.column_stack([relative, size[:, 0] / size[:, 1]]).astype(np.float32)


def hard_candidates(truth):
    output = []
    for (frame, cls), group in truth.groupby(['frame_id', 'class_name']):
        originals = group[BOX].to_numpy()
        for obj in originals:
            size = obj[2:] - obj[:2]
            for edge in range(4):
                for ratio in (.50, .57, .63, .72):
                    box = obj.copy()
                    box[edge] += (1 if edge < 2 else -1) * (1 - ratio) * size[edge % 2]
                    output.append((frame, cls, *box))
    hard = pd.DataFrame(output, columns=['frame_id', 'class_name', *BOX]).drop_duplicates().reset_index(drop=True)
    hard = hard[(hard.x2 - hard.x1 >= 4) & (hard.y2 - hard.y1 >= 4)].reset_index(drop=True)
    # Each proposal is an independent counterfactual, scored against every GT in the frame.
    hard['error'] = 0
    for (frame, cls), group in hard.groupby(['frame_id', 'class_name']):
        gt = truth[(truth.frame_id == frame) & (truth.class_name == cls)][BOX].to_numpy()
        hard.loc[group.index, 'error'] = (iou_matrix(group[BOX].to_numpy(), gt).max(1) < .6).astype(int)
    # Balance classes and days to keep the extra cache small and both training folds covered.
    strata = hard.groupby([hard.frame_id.str[-15:-7], 'class_name', 'error'])
    chosen = [group.sample(min(48, len(group)), random_state=213) for _, group in strata]
    return pd.concat(chosen, ignore_index=True).assign(part='synth')


def extraction(rows, mode):
    path = OUT / f'{mode}_tokens.pt'
    if path.exists():
        features = torch.load(path, weights_only=True, mmap=True)
        assert len(features) == len(rows)
        return features
    size = 140 if mode == 'lower' else 280
    enc.SIZE, enc.BATCH = size, 16 if mode == 'lower' else 12
    backbone = control.timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True,
                                         num_classes=0, img_size=size).eval().cuda()
    chunks = []
    for start in range(0, len(rows), 192):
        batch = rows.iloc[start:start+192].copy()
        chunk_path = OUT / f'{mode}_{size}_chunk_{start:04}.pt'
        if chunk_path.exists():
            chunk = torch.load(chunk_path, weights_only=True, mmap=True)
            assert len(chunk) == len(batch)
            chunks.append(chunk)
            continue
        if mode == 'lower':
            # Offset a same-size second view by half a box height, keeping the main view intact.
            height = batch.y2 - batch.y1
            shift = np.minimum(height * .5, (720 - batch.y2).clip(lower=0))
            batch.y1 += shift
            batch.y2 += shift
        result = enc.extract(backbone, batch)
        if mode == 'lower':
            grid = size // 14
            result = result[:, -(grid**2):].reshape(-1, grid, grid, 1024)
            result = result[:, grid//2:].float().mean((1, 2)).half()
        temporary = chunk_path.with_suffix('.tmp')
        torch.save(result, temporary)
        temporary.replace(chunk_path)
        chunks.append(result)
        print(f'{mode} extracted {min(start+192, len(rows))}/{len(rows)}', flush=True)
    del backbone
    torch.cuda.empty_cache()
    features = torch.cat(chunks)
    torch.save(features, path)
    return features


class FollowupHead(control.Refiner):
    def __init__(self, extra_dim, mixture=False):
        super().__init__('coarse')
        self.mixture = mixture
        if extra_dim:
            self.head[0] = nn.Linear(48 * 25 + 4 + extra_dim, 128)
        if mixture:
            self.head[-1] = nn.Linear(128, 17)  # 3 boxes with weights, plus existence and error logits


def train_seed(tokens, extra, cls, targets, overlaps, errors, weights, train, pred, seed, mode, hard_tokens):
    torch.manual_seed(seed)
    mixture = mode == 'mixture'
    net = FollowupHead(extra.shape[1], mixture).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.0006, weight_decay=.03)

    def predict(idx):
        if hard_tokens is not None and (idx >= len(tokens)).any():
            block = torch.empty((len(idx), *tokens.shape[1:]), dtype=tokens.dtype, device=tokens.device)
            original = idx < len(tokens)
            block[original] = tokens[idx[original]]
            block[~original] = hard_tokens[idx[~original] - len(tokens)]
        else:
            block = tokens[idx]
        return net(block.cuda(), torch.cat([cls[idx], extra[idx]], dim=1))

    for _ in range(15):
        net.train()
        for idx in train[torch.randperm(len(train))].split(48):
            logits = predict(idx)
            present = overlaps[idx] >= .2
            if mixture:
                hypotheses = logits[:, :15].reshape(-1, 3, 5)
                target = targets[idx, None, :].expand(-1, 3, -1)
                distances = F.smooth_l1_loss(hypotheses[..., :4], target, reduction='none').mean(-1)
                geom = -torch.logsumexp(F.log_softmax(hypotheses[..., 4], 1) - 20*distances, dim=1) / 20
                obj = F.binary_cross_entropy_with_logits(logits[:, 15], present.float(), reduction='none')
                err = F.binary_cross_entropy_with_logits(logits[:, 16], errors[idx], reduction='none')
                loss = weights[idx] * (4*geom*present + .4*obj + err)
            else:
                geom = F.smooth_l1_loss(logits[:, :4], targets[idx], reduction='none').mean(1)
                obj = F.binary_cross_entropy_with_logits(logits[:, 4], (overlaps[idx] > .2).float(), reduction='none')
                err = F.binary_cross_entropy_with_logits(logits[:, 5], errors[idx], reduction='none')
                loss = weights[idx] * (4*geom*present + .4*obj + err)
            opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            opt.step()
    net.eval()
    with torch.inference_mode():
        return torch.cat([predict(idx) for idx in pred.split(48)]).cpu().numpy()


def neural(mode, rows, truth, split_list):
    tokens = torch.load(WORK / 'tokens280.pt', weights_only=True, mmap=True)
    assert tokens.shape == (len(rows), 401, 1024)
    hard_tokens = None
    if mode == 'hard':
        hard = hard_candidates(truth)
        hard_tokens = extraction(hard, 'hard')
        rows = pd.concat([rows, hard], ignore_index=True)
    extra = crop_geometry(rows) if mode == 'geometry5' else np.empty((len(rows), 0), dtype=np.float32)
    if mode == 'lower':
        extra = extraction(rows, 'lower').float().numpy()
    tokens = tokens.cuda()
    if hard_tokens is not None:
        hard_tokens = hard_tokens.cuda()
    extra = torch.as_tensor(extra, device='cuda')
    cls = F.one_hot(torch.tensor(rows.class_name.map(CLASSES.index).to_numpy(), device='cuda'), 4).float()
    coords, iou = supervision(rows, truth)
    target, overlap, errors, weights = [torch.as_tensor(x, device='cuda', dtype=torch.float32) for x in
        (coords, iou, rows.error.fillna(0).to_numpy(), np.where(rows.part == 'synth', .30, 1.))]
    for split in split_list:
        if split == 'test' and all((OUT / f'{mode}_{day}.csv').exists() for day in DAYS):
            def ap(path):
                frame = pd.read_csv(path)
                return average_precision_score(frame.error, frame.score)

            if any(ap(OUT / f'{mode}_{day}.csv') <= ap(OUT / f'control_280_{day}.csv') for day in DAYS):
                print(mode, 'test training skipped: no gain on both folds', flush=True)
                continue
        train, pred = split_indices(rows, split)
        det = rows.iloc[pred].reset_index(drop=True)
        predictions = []
        for seed in SEEDS:
            path = OUT / f'{mode}_{split}_{seed}.npy'
            if path.exists():
                prediction = np.load(path)
            else:
                prediction = train_seed(tokens, extra, cls, target, overlap, errors, weights,
                                        torch.from_numpy(train), torch.from_numpy(pred), seed, mode, hard_tokens)
                np.save(path, prediction)
            predictions.append(prediction)
            print(mode, split, seed, 'ready', flush=True)
        base = base_for(det, split)
        if mode == 'mixture':
            signal = []
            for prediction in predictions:
                hypotheses = prediction[:, :15].reshape(-1, 3, 5)
                probabilities = torch.softmax(torch.from_numpy(hypotheses[..., 4]), 1).numpy()
                overlaps = np.column_stack([risk_parts(det, h)[0] for h in hypotheses.transpose(1, 0, 2)])
                existence = 1 / (1 + np.exp(-prediction[:, 15].clip(-40, 40)))
                signal.append(1 - existence * (probabilities * (overlaps >= .6)).sum(1))
            evidence = np.mean(signal, axis=0)
        else:
            evidence = control.geometry(det, np.mean(predictions, axis=0))
        save(det, split, {mode: rank_mix(det, base, evidence, .5, True)})


def save(det, split, scores):
    for name, score in scores.items():
        assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
        det[['id', 'frame_id', 'class_name', 'error']].assign(score=score).to_csv(OUT / f'{name}_{split}.csv', index=False)
        if split != 'test':
            print(split, name, f'{average_precision_score(det.error, score):.8f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='+', choices=['aggregation', 'geometry5', 'lower', 'hard', 'mixture'])
    parser.add_argument('--oof-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    rows, truth = inputs()
    splits = DAYS if args.oof_only else (*DAYS, 'test')
    for mode in args.mode:
        if mode == 'aggregation':
            for split in splits:
                _, pred = split_indices(rows, split)
                det = rows.iloc[pred].reset_index(drop=True)
                save(det, split, aggregation(det, split, base_for(det, split)))
        else:
            neural(mode, rows, truth, splits)


if __name__ == '__main__':
    main()
