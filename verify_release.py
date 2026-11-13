"""Verify real test coverage, saved-model reproduction and full submission merge."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from pipeline import ROOT,apply_scorer,digest,merge,save_json,validate_scores


def main():
    test=pd.read_csv(ROOT.parent/'data/csv/test.csv',dtype={'id':str},float_precision='round_trip')
    scores=pd.read_csv(ROOT/'outputs/person_scores.csv',dtype={'id':str},float_precision='round_trip')
    features=pd.read_csv(ROOT/'outputs/features.csv',dtype={'id':str},float_precision='round_trip')
    scorer=json.loads((ROOT/'outputs/scorer.json').read_text())
    expected=set(test.loc[test.class_name.eq('person'),'id'])
    validate_scores(scores,expected)
    tf=features[features.split.eq('test')].set_index('id').loc[scores.id].reset_index()
    calculated=apply_scorer(scorer,tf)
    max_difference=float(np.max(np.abs(calculated-scores.score)))
    assert max_difference<1e-12
    with tempfile.TemporaryDirectory(prefix='person_release_',dir=ROOT) as directory:
        work=Path(directory)
        baseline=pd.DataFrame({'id':test.id,'score':1-test.confidence}).sample(frac=1,random_state=42)
        baseline.to_csv(work/'base.csv',index=False,float_format='%.17g')
        merged_path=work/'merged.csv'
        merge(Namespace(data=ROOT.parent/'data',base=work/'base.csv',scores=ROOT/'outputs/person_scores.csv',destination=merged_path,mode='rank'))
        merged=pd.read_csv(merged_path,dtype={'id':str},float_precision='round_trip')
        validate_scores(merged,set(test.id))
        base=baseline.set_index('id').score
        actual=merged.set_index('id').score.reindex(base.index)
        person=base.index.isin(expected)
        assert np.array_equal(actual[~person].to_numpy(),base[~person].to_numpy())
        assert np.array_equal(np.sort(actual[person]),np.sort(base[person]))
        assigned=pd.DataFrame({'rank':scores.set_index('id').score,'assigned':actual}).loc[list(expected)].sort_values('rank')
        assert (np.diff(assigned.assigned.to_numpy())>=0).all()
    report={'person_rows':len(scores),'unique_person_ids':scores.id.nunique(),
            'complete_test_rows':len(test),'unchanged_other_rows':int((~person).sum()),
            'score_min':float(scores.score.min()),'score_max':float(scores.score.max()),
            'saved_scorer_max_abs_difference':max_difference,'rank_merge_preserves_person_distribution':True,
            'non_person_values_preserved_exactly':True,'person_scores_sha256':digest(ROOT/'outputs/person_scores.csv')}
    save_json(ROOT/'outputs/release_checks.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
