"""Warm CPU inference benchmark on the first 12 TRAIN person frames."""
import gc
import json
import os
import platform
import time
import numpy as np
import psutil
from PIL import Image
from pipeline import ROOT,Detector,Pose,read_data,save_json


def stats(values):
    return {'count':len(values),'median_seconds':float(np.median(values)),
            'mean_seconds':float(np.mean(values)),'total_seconds':float(sum(values))}


def main():
    data=read_data(ROOT.parent/'data')
    train=data[data.split.eq('train')]
    frames=sorted(train.frame_id.unique())[:12]
    images={f:np.asarray(Image.open(ROOT.parent/'data/imgs'/f'{f}.jpg').convert('RGB')) for f in frames}
    model=Detector(ROOT/'assets/deim_m.onnx',4)
    for _ in range(3):
        model(images[frames[0]])
    times=[]
    for frame in frames:
        start=time.perf_counter()
        model(images[frame])
        times.append(time.perf_counter()-start)
    detector_stats=stats(times)
    del model
    gc.collect()
    pose_path=ROOT/'assets/rtmpose_m26.onnx'
    if not pose_path.exists():
        pose_path=next((ROOT/'assets/pose_source').rglob('end2end.onnx'))
    model=Pose(pose_path,4)
    subset=train[train.frame_id.isin(frames)]
    times=[]
    for _,row in subset.iterrows():
        cached=json.loads((ROOT/'outputs/cache/pose'/f'{row.id}.json').read_text())
        window=np.asarray(cached['window'])
        if not times:
            for _ in range(3):
                model(images[row.frame_id],window)
        start=time.perf_counter()
        model(images[row.frame_id],window)
        times.append(time.perf_counter()-start)
    memory=psutil.Process().memory_info()
    report={'device':'CPUExecutionProvider','threads':4,'batch_size':1,'precision':'FP32',
            'logical_processors':os.cpu_count(),'processor':platform.processor(),
            'detector':detector_stats,'pose':stats(times),
            'peak_process_rss_mib':getattr(memory,'peak_wset',memory.rss)/2**20,
            'timing_scope':'per-model preprocessing + ONNX inference + decoding; excludes disk reads and model initialization'}
    save_json(ROOT/'outputs/benchmark.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
