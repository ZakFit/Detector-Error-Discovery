from precise_review import *
from scipy.optimize import linear_sum_assignment

def iou(a,b):
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(-1)
    return inter/np.maximum(1e-9,(a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter)

GROUPS=[[19,21],[24,25],[39,40],[46,48],[50,51],[71,72],[110,112],[122,123],
 [136,137],[145,146,147],[181,184],[191,193,195],[238,240,241],[248,252],
 [259,261],[278,279],[304,305,306],[312,313],[326,327],[333,335], [172,173],[204,206],[319,321]]

def read_manual():
    a=pd.concat([pd.read_csv(x) for x in sorted(OUT.glob('manual_[0-9]*.csv'))],ignore_index=True)
    assert len(a)==362 and a.review_number.nunique()==362
    revisions=OUT/'context_revisions.json'
    if revisions.exists():
        for rev in json.loads(revisions.read_text()):
            sel=a.review_number==rev['review_number']
            a.loc[sel,['x1p','y1p','x2p','y2p']]=rev['box'] if rev['box'] is not None else [np.nan]*4
            a.loc[sel,'quality']=rev['quality'];a.loc[sel,'note']=rev['note']
    a=a.merge(pd.read_csv(ROOT/'work/offline_review/index.csv'),on='review_number',validate='one_to_one')
    crops={x['review_number']:x['crop'] for x in json.loads((OUT/'crop_coordinates.json').read_text())}
    for k in BOX: a['manual_'+k]=np.nan
    for i,r in a.iterrows():
        c=crops[int(r.review_number)]
        bb=np.array([r.x1p,r.y1p,r.x2p,r.y2p])/100*np.array([c[2]-c[0],c[3]-c[1]]*2)+np.array(c[:2]*2)
        a.loc[i,['manual_'+k for k in BOX]]=np.clip(bb,[0,0,0,0],[1280,720,1280,720])
    override=OUT/'box_overrides.csv'
    if override.exists():
        for _,r in pd.read_csv(override).iterrows():
            sel=a.review_number==r.review_number
            for k in BOX: a.loc[sel,'manual_'+k]=r[k]
            a.loc[sel,'quality']=r.quality;a.loc[sel,'note']=r.note
    a['object_group']=a.review_number.astype(str)
    for group in GROUPS:
        a.loc[a.review_number.isin(group),'object_group']='-'.join(map(str,group))
    refs=[]
    for group,q in a[a.manual_x1.notna()].groupby('object_group',sort=False):
        assert q.frame_id.nunique()==1 and q.class_name.nunique()==1
        bb=np.median(q[['manual_'+k for k in BOX]].values,axis=0)
        quality=max(q.quality,key=lambda z:{'high':0,'medium':1,'low':2}[z])
        refs.append(dict(object_id=group,frame_id=q.frame_id.iloc[0],class_name=q.class_name.iloc[0],quality=quality,**dict(zip(BOX,bb))))
    refs=pd.DataFrame(refs)
    extra=OUT/'extra_objects.csv'
    if extra.exists():
        ex=pd.read_csv(extra).merge(a[['review_number','frame_id','class_name']],on='review_number')
        refs=pd.concat([refs,ex[['object_id','frame_id','class_name','quality']+BOX]],ignore_index=True)
    return a,refs

def match(det,refs):
    err=np.ones(len(det),int); best=np.zeros(len(det)); assigned=np.full(len(det),'',object)
    for (fid,cls),q in det.groupby(['frame_id','class_name']):
        g=refs[(refs.frame_id==fid)&(refs.class_name==cls)]
        if not len(g):continue
        m=iou(q[BOX],g[BOX]);valid=m>=.6
        ii,jj=linear_sum_assignment(np.where(valid,len(q)+1+m,0),maximize=True)
        good=valid[ii,jj]; idx=q.index.values
        err[idx[ii[good]]]=0;assigned[idx[ii[good]]]=g.object_id.values[jj[good]]
        best[idx]=m.max(axis=1)
    return err,best,assigned

def main():
    a,refs=read_manual()
    a['manual_error'],a['best_manual_iou'],a['matched_object']=match(a,refs)
    a.to_csv(OUT/'detection_audit.csv',index=False)
    refs.to_csv(OUT/'manual_boxes.csv',index=False)
    # Review every scored full frame with candidate and reconstructed person boxes.
    frames=sorted(a.frame_id.unique())
    for page in range((len(frames)+3)//4):
        sheet=Image.new('RGB',(1280,800),'#202126')
        for j,fid in enumerate(frames[page*4:page*4+4]):
            im=Image.open(D/f'{fid}.jpg').convert('RGB').resize((640,360)); dr=ImageDraw.Draw(im)
            q=a[(a.frame_id==fid)&(a.class_name=='person')]
            for _,r in q.iterrows():
                b=r[BOX].values.astype(float)/2;dr.rectangle(b.tolist(),outline='#30ff30',width=1)
                dr.text((b[0],b[1]-14),str(int(r.review_number)),font=SMALL,fill='#30ff30')
            for _,r in refs[(refs.frame_id==fid)&(refs.class_name=='person')].iterrows():
                b=r[BOX].values.astype(float)/2;dr.rectangle(b.tolist(),outline='#ff40ff',width=2)
            x=j%2*640;y=j//2*400;sheet.paste(im,(x,y+24))
            ImageDraw.Draw(sheet).text((x+3,y+3),fid[-15:]+' green=candidate purple=manual',font=FONT,fill='white')
        sheet.save(SHEETS/f'full_{page:02}.jpg',quality=94)
    # Independent reconstruction exercise: labels and GT stay hidden on these sheets.
    train=pd.read_csv(D/'train.csv');train=train[train.class_name.isin(['person','cup','nude_hands','tray'])]
    seen=pd.read_csv(OUT/'calibration_samples.csv').id
    blind=pd.concat([q.sample(min(6,len(q)),random_state=19) for _,q in train[~train.id.isin(seen)].groupby('class_name')]).reset_index(drop=True)
    blind['review_number']=np.arange(1,len(blind)+1)
    blind.to_csv(OUT/'blind_train_samples.csv',index=False)
    for page in range((len(blind)+5)//6):
        sheet=Image.new('RGB',(1320,920))
        for j,(_,r) in enumerate(blind.iloc[page*6:page*6+6].iterrows()):
            p,_=panel(r);sheet.paste(p,((j%3)*440,(j//3)*460))
        sheet.save(SHEETS/f'blind_{page:02}.jpg',quality=94)
    base=pd.read_csv(ROOT/'outputs/submission_person_refiner_280.csv').set_index('id').score
    a['baseline']=a.id.map(base)
    a['disagreement']=np.abs(a.manual_error-a.baseline)
    print('records',len(a),'objects',len(refs),'frames',len(frames))
    print(a.groupby('class_name').manual_error.agg(['count','sum']))
    print('largest disagreements\n',a.sort_values('disagreement',ascending=False)[['review_number','class_name','manual_error','best_manual_iou','quality','baseline']].head(24).round(3).to_string(index=False))
    for (fid,cls),q in refs.groupby(['frame_id','class_name']):
        if len(q)<2:continue
        m=iou(q[BOX],q[BOX]);ii,jj=np.where(np.triu(m,1)>.75)
        for i,j in zip(ii,jj): print('possible same object',q.object_id.iloc[i],q.object_id.iloc[j],round(m[i,j],3))

if __name__=='__main__':main()
