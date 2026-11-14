"""Export template-ordered CSVs and quantify changes relative to the Public 0.93578 file."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / 'outputs'
OUT = OUTPUTS / 'followup_280'
AUDIT = OUTPUTS / 'manual_bbox_review'
DAYS = ('20261025', '20261026')


def export(name, scores, template, baseline):
    result = scores.reindex(template)
    assert len(result) == 362 and result.index.is_unique and result.between(0, 1).all()
    path = OUT / f'submission_{name}.csv'
    result.rename('score').to_csv(path)
    reloaded = pd.read_csv(path, dtype={'id': str})
    assert reloaded.columns.tolist() == ['id', 'score'] and reloaded.id.tolist() == template.tolist()
    assert reloaded.score.between(0, 1).all()
    return {'file': path.name, 'changed_vs_093578': int((abs(result-baseline) > 1e-10).sum()),
            'top50_overlap_vs_093578': len(set(result.nlargest(50).index) & set(baseline.nlargest(50).index))}


def main():
    template = pd.Index(pd.read_csv(AUDIT / 'source_tables/sample_submission.csv', dtype={'id': str}).id, name='id')
    baseline = pd.read_csv(OUTPUTS / 'submission_manual_review_full.csv').set_index('id').score.reindex(template)
    control = pd.read_csv(OUT / 'control_280_test.csv').set_index('id').score.reindex(template)
    old_public = pd.read_csv(OUTPUTS / 'submission_person_refiner_280.csv').set_index('id').score.reindex(template)
    np.testing.assert_allclose(control, old_public, rtol=0, atol=1e-12)
    audit = pd.read_csv(AUDIT / 'detection_audit.csv').set_index('id').reindex(template)
    manual = .85*audit.boundary_failure_fraction+.15*audit.manual_error_nominal
    weights = audit.quality.map({'high': .22, 'medium': .12, 'low': .05}) * (.5 + abs(manual-.5))
    weights[audit.conservative_override & audit.error_reason.eq('no_same_class_object')] = .75
    weights[audit.conservative_override & ~audit.error_reason.eq('no_same_class_object')] = .55
    np.testing.assert_allclose(baseline, control + weights*(manual-control), atol=1e-10, rtol=0)

    metrics, candidates = [], []
    for path in sorted(OUT.glob(f'*_{DAYS[0]}.csv')):
        name = path.name.removesuffix(f'_{DAYS[0]}.csv')
        if name in ('annotation_style', 'control_280') or name.startswith('submission_'):
            continue
        if any(not (OUT / f'{name}_{day}.csv').exists() for day in DAYS):
            continue
        row = {'name': name}
        for day in DAYS:
            data = pd.read_csv(OUT / f'{name}_{day}.csv')
            origin = pd.read_csv(OUT / f'control_280_{day}.csv')
            assert data.id.tolist() == origin.id.tolist() and data.error.equals(origin.error)
            row[day] = average_precision_score(data.error, data.score)
            row[f'delta_{day}'] = row[day] - average_precision_score(origin.error, origin.score)
        row['mean'] = np.mean([row[day] for day in DAYS])
        metrics.append(row)
        test_path = OUT / f'{name}_test.csv'
        if test_path.exists() and all(row[f'delta_{day}'] > 0 for day in DAYS):
            score = pd.read_csv(test_path).set_index('id').score.reindex(template)
            combined = (baseline + .5*(1-weights)*(score-control)).clip(0,1)
            candidates.append(export(f'manual_plus_{name}', combined, template, baseline))

    decisions = pd.read_csv(OUT / 'targeted_decisions.csv')
    cases = pd.read_csv(OUT / 'targeted_cases.csv')
    assert set(cases.review_number) == set(decisions.review_number)
    edits = decisions.merge(cases[['review_number','id']], on='review_number', validate='one_to_one')
    accepted = edits.loc[edits.decision.eq('correct'), 'id']
    reviewed = baseline.copy()
    reviewed.loc[accepted] *= .70
    candidates.append(export('manual_targeted_review', reviewed, template, baseline))

    # Two boxes around visibly the same hand. Preserve the model/geometry's existing winner.
    pair = audit[audit.review_number.isin([259, 261])]
    assert len(pair) == 2 and pair.object_group.nunique() == 1 and pair.frame_id.nunique() == 1
    weaker = baseline.loc[pair.index].idxmax()
    reviewed.loc[weaker] += .25 * (1 - reviewed.loc[weaker])
    candidates.append(export('manual_targeted_plus_duplicate', reviewed, template, baseline))

    # Redistribute a group constraint only where a plausible winner also agrees with the baseline.
    group_review = reviewed.copy()
    grouped = audit[audit.object_group.astype(str).str.contains('-')]
    for _, group in grouped.groupby('object_group'):
        if (group.quality == 'low').any() or len(group) < 2 or weaker in group.index:
            continue
        winner = group.best_manual_iou.idxmax()
        if winner != baseline.loc[group.index].idxmin() or group.at[winner, 'best_manual_iou'] < .66:
            continue
        for other in group.index.drop(winner):
            margin = group.at[winner, 'best_manual_iou'] - group.at[other, 'best_manual_iou']
            if margin >= .12:
                group_review.loc[other] += .20 * (1 - group_review.loc[other])
    candidates.append(export('manual_group_constraints', group_review, template, baseline))
    pd.DataFrame(metrics).sort_values('mean', ascending=False).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(candidates).to_csv(OUT/'candidates.csv', index=False)
    print(pd.DataFrame(metrics).round(6).to_string(index=False))
    print(pd.DataFrame(candidates).to_string(index=False))


if __name__ == '__main__':
    main()
