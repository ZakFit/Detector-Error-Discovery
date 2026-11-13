"""CPU person-error scoring: frozen DEIMv2 + RTMPose, day-held-out evaluation.

Run from the workspace with: python person_pipeline/pipeline.py --help
Only the supplied train error labels are used for fitting and model selection.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
COORDS = ['x1','y1','x2','y2']
VERSION = 'person-v1'
DETECTOR_FEATURES = ['confidence','det_iou','det_score','bottom_delta','top_delta','log_area_ratio']
POSE_FEATURES = ['pose_box_iou','pose_quality','pose_outside','feet_missing']


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ious(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1,4)
    if not len(boxes):
        return np.empty(0)
    box = np.asarray(box,dtype=float)
    intersection = np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])).prod(1)
    area = np.maximum(0,box[2:]-box[:2]).prod()
    other = np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
    return intersection / np.maximum(area+other-intersection,1e-9)


def read_data(data):
    parts = []
    for split in ['train','test']:
        frame = pd.read_csv(data/'csv'/f'{split}.csv', dtype={'id':str,'frame_id':str},float_precision='round_trip')
        frame = frame.loc[frame.class_name.eq('person')].copy()
        if frame.id.duplicated().any():
            raise ValueError('Duplicate person ids in '+split)
        if not np.isfinite(frame[COORDS+['confidence']].to_numpy(float)).all():
            raise ValueError('Nonfinite input features')
        if ((frame.x2<=frame.x1)|(frame.y2<=frame.y1)).any():
            raise ValueError('Invalid input box')
        frame['split'] = split
        frame['day'] = frame.frame_id.str.rsplit('_',n=2).str[-2]
        if split == 'test':
            frame['error'] = np.nan
        parts.append(frame)
    return pd.concat(parts,ignore_index=True)


def ort_session(model, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])


class Detector:
    def __init__(self, model, threads):
        self.session = ort_session(model,threads)

    def __call__(self, image):
        im = Image.fromarray(image)
        value = np.asarray(im.resize((640,640),Image.Resampling.BILINEAR),dtype=np.float32)/255.
        value = (value-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        labels,boxes,scores = self.session.run(None,{
            'images':value.transpose(2,0,1)[None].copy(),
            'orig_target_sizes':np.array([[im.width,im.height]],np.int64)})
        selected = (labels[0]==0)&(scores[0]>=.05)
        boxes,scores = boxes[0][selected],scores[0][selected]
        boxes[:,[0,2]] = boxes[:,[0,2]].clip(0,im.width)
        boxes[:,[1,3]] = boxes[:,[1,3]].clip(0,im.height)
        valid = (boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
        return {'boxes':boxes[valid].tolist(),'scores':scores[valid].tolist()}


def choose_box(original, detection):
    boxes = np.asarray(detection['boxes'],float).reshape(-1,4)
    scores = np.asarray(detection['scores'],float)
    keep = scores>=.15
    boxes,scores = boxes[keep],scores[keep]
    overlap = ious(original,boxes)
    if not len(overlap) or overlap.max()<.05:
        return None,0.,0.,0.
    order = np.argsort(-overlap,kind='stable')
    k = order[0]
    second = float(overlap[order[1]]) if len(order)>1 else 0.
    return boxes[k],float(overlap[k]),float(scores[k]),second


def pose_window(original, matched, width, height):
    # The target's original bottom must not truncate the evidence used to judge it.
    base = np.asarray(original,float).copy()
    if matched is not None:
        base[:2] = np.minimum(base[:2],matched[:2])
        base[2:] = np.maximum(base[2:],matched[2:])
    w,h = base[2:]-base[:2]
    base += np.array([-.10*w,-.05*h,.10*w,.25*h])
    base[[0,2]] = base[[0,2]].clip(0,width)
    base[[1,3]] = base[[1,3]].clip(0,height)
    return base


def pose_transform(box):
    """MMPose top-down affine, rotation=0, Halpe26 192x256, padding=1.25."""
    center = (box[:2]+box[2:])/2
    scale = (box[2:]-box[:2])*1.25
    if scale[0] > scale[1]*.75:
        scale[1] = scale[0]/.75
    else:
        scale[0] = scale[1]*.75
    matrix = np.array([[192/scale[0],0,96-center[0]*192/scale[0]],
                       [0,256/scale[1],128-center[1]*256/scale[1]]],np.float32)
    return matrix,center,scale


class Pose:
    def __init__(self, model, threads):
        self.session = ort_session(model,threads)
        self.name = self.session.get_inputs()[0].name

    def __call__(self, image, box):
        import cv2
        matrix,center,scale = pose_transform(box)
        crop = cv2.warpAffine(image,matrix,(192,256),flags=cv2.INTER_LINEAR)
        value = (crop.astype(np.float32)-np.array([123.675,116.28,103.53],np.float32))/np.array([58.395,57.12,57.375],np.float32)
        output = self.session.run(None,{self.name:value.transpose(2,0,1)[None].copy()})
        sx = next(x for x in output if x.shape[-1]==384)[0]
        sy = next(x for x in output if x.shape[-1]==512)[0]
        keypoints = np.stack([sx.argmax(-1),sy.argmax(-1)],axis=-1).astype(float)/2
        keypoints = keypoints/np.array([192,256])*scale+center-scale/2
        scores = np.minimum(sx.max(-1),sy.max(-1))
        if keypoints.shape != (26,2):
            raise ValueError('Expected Halpe26 pose output')
        return {'keypoints':keypoints.tolist(),'scores':scores.tolist(),'window':box.tolist()}


def make_features(row, detection, pose):
    box = row[COORDS].to_numpy(float)
    w,h = box[2:]-box[:2]
    match,overlap,confidence,second = choose_box(box,detection)
    features = {'confidence':float(row.confidence),'det_iou':overlap,'det_score':confidence,
                'det_missing':float(match is None),'second_iou':second,
                'bottom_delta':0.,'top_delta':0.,'log_area_ratio':0.,'side_delta':0.}
    if match is not None:
        features.update(bottom_delta=float(np.clip((match[3]-box[3])/h,-3,3)),
                        top_delta=float(np.clip((match[1]-box[1])/h,-3,3)),
                        log_area_ratio=float(np.clip(np.log((match[2]-match[0])*(match[3]-match[1])/(w*h)),-3,3)),
                        side_delta=float((abs(match[0]-box[0])+abs(match[2]-box[2]))/(2*w)))
    points = np.asarray(pose['keypoints'],float)
    scores = np.clip(np.asarray(pose['scores'],float),0,1)
    valid = scores>=.3
    outside = ((points[:,0]<box[0])|(points[:,0]>box[2])|(points[:,1]<box[1])|(points[:,1]>box[3]))
    feet = np.array([15,16,20,21,22,23,24,25])
    valid_feet = feet[valid[feet]]
    features.update(pose_quality=float(scores.mean()),pose_valid=float(valid.mean()),
                    pose_torso_quality=float(scores[[5,6,11,12]].mean()),
                    pose_outside=float((outside*valid*scores).sum()/max((valid*scores).sum(),1e-9)),
                    feet_missing=float(len(valid_feet)==0),
                    feet_delta=float(np.clip((points[valid_feet,1].max()-box[3])/h,-3,3)) if len(valid_feet) else 0.,
                    pose_box_iou=0.)
    if valid.sum()>=4:
        lo,hi = points[valid].min(0),points[valid].max(0)
        margin = (hi-lo)*.05
        features['pose_box_iou'] = float(ious(box,np.r_[lo-margin,hi+margin][None])[0])
    return features


def load_cache(path, fingerprint):
    if path.exists():
        content = json.loads(path.read_text(encoding='utf-8'))
        if content.get('fingerprint') == fingerprint:
            return content
    return None


def extract(args):
    data = read_data(args.data)
    if args.split != 'all':
        data = data[data.split.eq(args.split)]
    frames = sorted(data.frame_id.unique())
    if args.limit_frames:
        frames = frames[:args.limit_frames]
        data = data[data.frame_id.isin(frames)]
    detector_path = args.assets/'deim_m.onnx'
    pose_path = args.assets/'rtmpose_m26.onnx'
    if not pose_path.exists():
        pose_path = next((args.assets/'pose_source').rglob('end2end.onnx'))
    detector_hash,pose_hash = digest(detector_path),digest(pose_path)
    cache = args.output/'cache'
    detections,images = {},{}
    detector = None
    for k,frame in enumerate(frames):
        path = args.data/'imgs'/(frame+'.jpg')
        image_hash = digest(path)
        images[frame] = image_hash
        fingerprint = [VERSION,detector_hash,image_hash]
        target = cache/'detector'/(frame+'.json')
        result = load_cache(target,fingerprint)
        if result is None:
            if detector is None:
                detector = Detector(detector_path,args.threads)
            array = np.asarray(Image.open(path).convert('RGB'))
            started = time.perf_counter()
            result = detector(array)
            result.update(fingerprint=fingerprint,seconds=time.perf_counter()-started,
                          image_size=[int(array.shape[1]),int(array.shape[0])])
            save_json(target,result)
        detections[frame] = result
        if k%10==0 or k==len(frames)-1:
            print('Detector',k+1,'/',len(frames),flush=True)
    del detector
    gc.collect()
    pose_model = None
    rows = []
    for frame, group in data.groupby('frame_id',sort=True):
        array = None
        detection = detections[frame]
        for _,row in group.iterrows():
            original = row[COORDS].to_numpy(float)
            match,*_ = choose_box(original,detection)
            width,height = detection['image_size']
            window = pose_window(original,match,width,height)
            fingerprint = [VERSION,pose_hash,images[frame],window.tolist()]
            target = cache/'pose'/(row.id+'.json')
            pose = load_cache(target,fingerprint)
            if pose is None:
                if pose_model is None:
                    pose_model = Pose(pose_path,args.threads)
                if array is None:
                    array = np.asarray(Image.open(args.data/'imgs'/(frame+'.jpg')).convert('RGB'))
                started = time.perf_counter()
                pose = pose_model(array,window)
                pose.update(fingerprint=fingerprint,seconds=time.perf_counter()-started)
                save_json(target,pose)
            values = {key:row[key] for key in ['id','frame_id','day','split','error']}
            values.update(make_features(row,detection,pose))
            rows.append(values)
        print('Pose',len(rows),'/',len(data),flush=True)
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    filename = 'features.csv' if args.split=='all' and not args.limit_frames else f'features_{args.split}.csv'
    result.to_csv(args.output/filename,index=False,float_format='%.17g')
    save_json((args.output/filename).with_suffix('.meta.json'),{
        'feature_version':VERSION,'models':{'deim_m':detector_hash,'rtmpose_m26':pose_hash},
        'inputs':{split:digest(args.data/'csv'/f'{split}.csv') for split in ['train','test']},
        'rows':len(result),'frames':len(frames)})
    print('Features saved:',len(result),flush=True)


def model_for(name):
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,solver='lbfgs',random_state=2026))


def columns_for(name):
    return DETECTOR_FEATURES + (POSE_FEATURES if name=='detector_pose_lr' else [])


def fit_predict(name, training, validation):
    if name=='confidence':
        return 1-validation.confidence.to_numpy(),None
    if name=='detector_iou':
        return 1-validation.det_iou.to_numpy(),None
    columns = columns_for(name)
    model = model_for(name)
    model.fit(training[columns],training.error.astype(int))
    return model.predict_proba(validation[columns])[:,1],model


def rank_transfer(new_scores, old_scores):
    """Keep the base person's score distribution, changing only its ordering."""
    new_scores,old_scores = np.asarray(new_scores),np.asarray(old_scores)
    if len(np.unique(new_scores))<=1:
        return old_scores.copy()
    result = np.empty(len(new_scores))
    order = np.argsort(new_scores,kind='stable')
    result[order] = np.sort(old_scores,kind='stable')
    return result


def evaluate(args):
    metadata=json.loads((args.output/'features.meta.json').read_text())
    if metadata['feature_version']!=VERSION or any(metadata['inputs'][s]!=digest(args.data/'csv'/f'{s}.csv') for s in ['train','test']):
        raise ValueError('Inputs/feature code changed; rerun extract')
    f = pd.read_csv(args.output/'features.csv',dtype={'id':str,'day':str,'frame_id':str},float_precision='round_trip')
    expected = read_data(args.data)
    if set(f.id)!=set(expected.id) or f.id.duplicated().any():
        raise ValueError('Features must cover every train/test person exactly once')
    train,test = f[f.split.eq('train')].copy(),f[f.split.eq('test')].copy()
    train=train.sort_values('id').reset_index(drop=True)
    test=test.sort_values('id').reset_index(drop=True)
    if not np.isfinite(f[DETECTOR_FEATURES+POSE_FEATURES].to_numpy()).all():
        raise ValueError('Invalid numeric features')
    days=sorted(train.day.unique())
    if len(days)<2:
        raise ValueError('Need at least two UTC days for validation')
    full = pd.read_csv(args.data/'csv'/'train.csv',dtype={'id':str},float_precision='round_trip')
    full = full[full.class_name.ne('table')].copy()
    full['day']=full.frame_id.str.rsplit('_',n=2).str[-2]
    report={'validation':'leave-one-UTC-day-out; model choice uses these same small validation folds',
            'non_person_baseline':'1-confidence (team reannotate_mean predictions not supplied)',
            'candidates':{},'train_person_rows':len(train),'test_person_rows':len(test)}
    oof=train[['id','frame_id','day','error']].copy()
    for name in ['confidence','detector_iou','detector_lr','detector_pose_lr']:
        predictions=np.zeros(len(train))
        scores=[]
        for day in days:
            mask=train.day.eq(day)
            held=train[mask]
            predictions[mask],_=fit_predict(name,train[~mask],held)
            metric={'day':day,'train_days':sorted(train.loc[~mask,'day'].unique()),
                    'training_rows':int((~mask).sum()),'validation_rows':int(mask.sum()),
                    'person_ap':float(average_precision_score(held.error,predictions[mask]))}
            all_day=full[full.day.eq(day)].copy()
            base=(1-all_day.confidence).to_numpy()
            index=all_day.id.isin(held.id)
            mapping=dict(zip(held.id,predictions[mask]))
            aligned=all_day.loc[index,'id'].map(mapping).to_numpy()
            direct=base.copy()
            direct[index]=aligned
            reranked=base.copy()
            reranked[index]=rank_transfer(aligned,base[index])
            metric['overall_ap_direct']=float(average_precision_score(all_day.error,direct))
            metric['overall_ap_rank']=float(average_precision_score(all_day.error,reranked))
            metric['person_ap_after_rank_merge']=float(average_precision_score(all_day.loc[index,'error'],reranked[index]))
            scores.append(metric)
        report['candidates'][name]={'folds':scores,'mean_person_ap':float(np.mean([s['person_ap'] for s in scores])),
                                    'min_person_ap':float(min(s['person_ap'] for s in scores)),
                                    'pooled_person_ap':float(average_precision_score(train.error,predictions))}
        oof[name]=predictions
    baseline=report['candidates']['confidence']
    # Prefer reproducible gains on BOTH days; avoid selecting only a large easy-day gain.
    eligible=[name for name,item in report['candidates'].items()
              if all(a['person_ap']>=b['person_ap']-1e-12 for a,b in zip(item['folds'],baseline['folds']))]
    selected=max(eligible,key=lambda name:(report['candidates'][name]['min_person_ap'],report['candidates'][name]['mean_person_ap']))
    predictions,model=fit_predict(selected,train,test)
    report['selected']=selected
    report['selection_rule']='non-worse than confidence on each day, then maximum worst-day person AP, then mean AP'
    report['feature_columns']=columns_for(selected) if model is not None else [selected]
    if model is not None:
        scaler=model.named_steps['standardscaler']
        lr=model.named_steps['logisticregression']
        exported={'kind':selected,'features':columns_for(selected),'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                  'coefficients':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),'error_class':1}
    else:
        exported={'kind':selected,'error_class':1}
    exported['feature_version']=VERSION
    exported['models']=metadata['models']
    save_json(args.output/'scorer.json',exported)
    reproduced=apply_scorer(exported,test)
    if not np.allclose(reproduced,predictions,rtol=1e-12,atol=1e-12):
        raise ValueError('Saved scorer does not reproduce fitted model')
    save_json(args.output/'validation.json',report)
    oof.to_csv(args.output/'validation_oof.csv',index=False,float_format='%.17g')
    scores=pd.DataFrame({'id':test.id,'score':predictions})
    validate_scores(scores,set(expected.loc[expected.split.eq('test'),'id']))
    scores.to_csv(args.output/'person_scores.csv',index=False,float_format='%.17g')
    print(json.dumps(report,indent=2),flush=True)


def apply_scorer(scorer, features):
    if scorer['kind']=='confidence':
        return 1-features.confidence.to_numpy()
    if scorer['kind']=='detector_iou':
        return 1-features.det_iou.to_numpy()
    values=features[scorer['features']].to_numpy(float)
    values=(values-np.asarray(scorer['mean']))/np.asarray(scorer['scale'])
    logits=values@np.asarray(scorer['coefficients'])+scorer['intercept']
    return 1/(1+np.exp(-np.clip(logits,-700,700)))


def predict(args):
    source=args.output/'features_test.csv'
    if not source.exists():
        source=args.output/'features.csv'
    features=pd.read_csv(source,dtype={'id':str},float_precision='round_trip')
    features=features[features.split.eq('test')].copy()
    scorer=json.loads((args.output/'scorer.json').read_text(encoding='utf-8'))
    metadata=json.loads(source.with_suffix('.meta.json').read_text())
    if scorer['feature_version']!=metadata['feature_version'] or scorer['models']!=metadata['models']:
        raise ValueError('Features are incompatible with the saved scorer')
    if metadata['inputs']['test']!=digest(args.data/'csv'/'test.csv'):
        raise ValueError('Test detections changed; rerun extract')
    scores=pd.DataFrame({'id':features.id,'score':apply_scorer(scorer,features)})
    expected=pd.read_csv(args.data/'csv'/'test.csv',dtype={'id':str})
    validate_scores(scores,set(expected.loc[expected.class_name.eq('person'),'id']))
    scores.to_csv(args.output/'person_scores.csv',index=False,float_format='%.17g')
    print('Predicted:',len(scores),'person scores',flush=True)


def validate_scores(scores, expected_ids):
    if set(scores.columns)!= {'id','score'} or scores.id.duplicated().any() or set(scores.id)!=set(expected_ids):
        raise ValueError('Scores must have id,score and cover the expected ids exactly once')
    values=scores.score.to_numpy(float)
    if not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
        raise ValueError('Scores must be finite values in [0,1]')


def merge(args):
    base=pd.read_csv(args.base,dtype={'id':str},float_precision='round_trip')
    test=pd.read_csv(args.data/'csv'/'test.csv',dtype={'id':str})
    person=pd.read_csv(args.scores,dtype={'id':str},float_precision='round_trip')
    validate_scores(base,set(test.id))
    validate_scores(person,set(test.loc[test.class_name.eq('person'),'id']))
    indices=base.id.isin(person.id)
    aligned=base.loc[indices,'id'].map(person.set_index('id').score).to_numpy()
    original=base.loc[~indices,'score'].copy()
    if args.mode=='rank':
        aligned=rank_transfer(aligned,base.loc[indices,'score'].to_numpy())
    base.loc[indices,'score']=aligned
    assert original.equals(base.loc[~indices,'score'])
    validate_scores(base,set(test.id))
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    base.to_csv(args.destination,index=False,float_format='%.17g')
    print('Merged:',int(indices.sum()),'person rows;',len(base),'total; mode='+args.mode,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'data')
    parser.add_argument('--assets',type=Path,default=ROOT/'assets')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    sub=parser.add_subparsers(dest='command',required=True)
    ex=sub.add_parser('extract')
    ex.add_argument('--threads',type=int,default=4)
    ex.add_argument('--split',choices=['train','test','all'],default='all')
    ex.add_argument('--limit-frames',type=int,default=0)
    sub.add_parser('evaluate')
    sub.add_parser('predict')
    me=sub.add_parser('merge')
    me.add_argument('--base',type=Path,required=True)
    me.add_argument('--scores',type=Path,default=ROOT/'outputs'/'person_scores.csv')
    me.add_argument('--destination',type=Path,required=True)
    me.add_argument('--mode',choices=['direct','rank'],default='rank',
                    help='rank preserves base person-score values; direct replaces their scale (default: rank)')
    args=parser.parse_args()
    {'extract':extract,'evaluate':evaluate,'predict':predict,'merge':merge}[args.command](args)


if __name__=='__main__':
    main()
