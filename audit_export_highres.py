from pathlib import Path
import pandas as pd,numpy as np
from sklearn.metrics import average_precision_score as ap
from scipy.stats import spearmanr
p=Path('data/gpu_runs/highres_refiner')
out=Path('../../outputs')
col='coarse_rank_0.5'
d=pd.read_csv(p/'oof.csv')
rng=np.random.default_rng(715)
for day,g in d.groupby('day'):
 groups=[q.index.to_numpy() for _,q in g.groupby('frame_id')]
 delta=[]
 for draw in rng.integers(len(groups),size=(1000,len(groups))):
  q=d.loc[np.concatenate([groups[i] for i in draw])]
  if q.error.nunique()==2:
   delta.append(ap(q.error,q[col])-ap(q.error,q.baseline))
 print(day,'delta CI',np.quantile(delta,[.025,.975]))
q=pd.read_csv(p/'test.csv')
template=pd.read_csv('data/raw/sample_submission.csv')[['id']]
s=template.merge(q[['id',col]],on='id',validate='one_to_one').rename(columns={col:'score'})
b=pd.read_csv(out/'submission_refinement_person_geometry_half.csv')
assert len(s)==len(b)==362 and s.id.equals(b.id) and s.score.between(0,1).all()
assert np.allclose(q.loc[q.class_name!='person',col],q.loc[q.class_name!='person','baseline'],atol=1e-12,rtol=0)
s.to_csv(out/'submission_person_refiner_280.csv',index=False)
print('changed',sum(abs(s.score-b.score)>1e-12),'spearman',spearmanr(s.score,b.score).statistic)
print('top50 overlap',len(set(s.nlargest(50,'score').id)&set(b.nlargest(50,'score').id)))
