"""Integration/geometry checks independent of model downloads."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pipeline import ious, merge, pose_transform, rank_transfer, validate_scores


class PipelineChecks(unittest.TestCase):
    def test_geometry_and_coordinate_order(self):
        box=np.array([10.,20.,110.,220.])
        np.testing.assert_allclose(ious(box,[[10,20,110,220],[10,20,110,120],[110,20,210,220]]),[1,.5,0])
        matrix,center,scale=pose_transform(box)
        np.testing.assert_allclose(matrix@np.r_[center,1],[96,128],atol=1e-5)
        np.testing.assert_allclose(matrix@np.r_[center-scale/2,1],[0,0],atol=1e-5)
        np.testing.assert_allclose(matrix@np.r_[center+scale/2,1],[192,256],atol=1e-5)

    def test_invalid_scores_are_rejected(self):
        cases=[pd.DataFrame({'id':['a','a'],'score':[.1,.2]}),
               pd.DataFrame({'id':['b'],'score':[.2]}),
               pd.DataFrame({'id':['a'],'score':[np.nan]}),
               pd.DataFrame({'id':['a'],'score':[1.01]})]
        for case in cases:
            with self.assertRaises(ValueError):
                validate_scores(case,{'a'})

    def test_merge_uses_ids_and_preserves_other_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'csv').mkdir()
            pd.DataFrame({'id':['p1','hand','p2'],'class_name':['person','nude_hands','person']}).to_csv(p/'csv'/'test.csv',index=False)
            pd.DataFrame({'id':['hand','p2','p1'],'score':[.765,.1,.2]}).to_csv(p/'base.csv',index=False)
            pd.DataFrame({'id':['p1','p2'],'score':[.9,.3]}).to_csv(p/'scores.csv',index=False)
            merge(Namespace(data=p,base=p/'base.csv',scores=p/'scores.csv',destination=p/'merged.csv',mode='direct'))
            result=pd.read_csv(p/'merged.csv').set_index('id').score
            self.assertAlmostEqual(result['hand'],.765)
            self.assertAlmostEqual(result['p1'],.9)
            self.assertAlmostEqual(result['p2'],.3)

    def test_rank_merge_preserves_base_distribution(self):
        result=rank_transfer([.9,.1,.5],[.2,.3,.1])
        np.testing.assert_allclose(result,[.3,.1,.2])
        np.testing.assert_allclose(np.sort(result),[.1,.2,.3])


if __name__=='__main__':
    unittest.main()
