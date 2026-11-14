"""Export only changes supported by a second visual pass."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score as ap

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'outputs/followup_280'
DATA = ROOT/'outputs/manual_bbox_review'


def main():
    baseline = pd.read_csv(ROOT/'outputs/submission_manual_review_full.csv', dtype={'id':str})
    template = pd.read_csv(DATA/'source_tables/sample_submission.csv', dtype={'id':str})
    pd.testing.assert_series_equal(baseline.id, template.id)
    table = pd.read_csv(OUT/'context_validation.csv', dtype={'id':str})
    test = table[table.split.eq('test')].set_index('id')
    audit = pd.read_csv(DATA/'detection_audit.csv', dtype={'id':str}).set_index('review_number')
    original = baseline.set_index('id').score
    plate_ids = audit.loc[[73,311,317,324,330,339], 'id']
    duplicate_ids = audit.loc[[137,173], 'id']
    assert len(original)==362 and original.index.is_unique and set(original.index)==set(test.index)
    assert test.loc[plate_ids,'background_distance'].le(24).all()
    just_plates = original.copy()
    just_plates.loc[plate_ids] += .35*(1-just_plates.loc[plate_ids])
    joint = just_plates.copy()
    assert test.loc[duplicate_ids,'joint'].gt(test.loc[duplicate_ids,'independent']).all()
    joint.loc[duplicate_ids] += .5*(test.loc[duplicate_ids,'joint']-test.loc[duplicate_ids,'independent'])
    frame = pd.DataFrame({'before':original,'plate_review':just_plates,'plate_and_joint':joint})
    changed = frame[frame.before.ne(frame.plate_and_joint)].join(audit.reset_index().set_index('id')[['review_number','note']])
    changed.to_csv(OUT/'context_manual_changes.csv')
    for name, score in [('context_plate_review',just_plates), ('context_plate_and_joint',joint)]:
        path=OUT/f'submission_manual_{name}.csv'
        score.rename('score').to_csv(path)
        check=pd.read_csv(path,dtype={'id':str})
        assert check.columns.tolist()==['id','score'] and check.id.equals(template.id) and check.score.between(0,1).all()
        print(name, 'changed',int(score.ne(original).sum()))
    records=[]
    for split,q in table[table.split.ne('test')].groupby('split'):
        for name in ('score','temporal','background','background_relaxed','independent','joint'):
            records.append(dict(day=split,method=name,ap=ap(q.error,q[name]),detections=len(q),frames=q.frame_id.nunique()))
    pd.DataFrame(records).to_csv(OUT/'context_metrics.csv',index=False)


if __name__=='__main__':
    main()
