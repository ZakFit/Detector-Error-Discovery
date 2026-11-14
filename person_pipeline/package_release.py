"""Package only deliverable code, model artifacts and calculated outputs."""
import json
import shutil
import zipfile
from pipeline import ROOT,digest,save_json


def main():
    assets=ROOT/'assets'
    if not (assets/'rtmpose_m26.onnx').exists():
        pose=next((assets/'pose_source').rglob('end2end.onnx'))
        shutil.copy2(pose,assets/'rtmpose_m26.onnx')
    if not (assets/'DEIMV2_LICENSE.md').exists():
        license_path=next((assets/'deim_source').rglob('LICENSE.md'))
        shutil.copy2(license_path,assets/'DEIMV2_LICENSE.md')
    models={name:{'sha256':digest(assets/name),'bytes':(assets/name).stat().st_size}
            for name in ['deim_m.onnx','rtmpose_m26.onnx']}
    save_json(assets/'model_manifest.json',{'models':models,'device':'CPU','batch_size':1,'precision':'FP32'})
    files=[ROOT/name for name in ['README.md','pipeline.py','setup_assets.py','export_detector.py',
                                  'test_pipeline.py','verify_release.py','benchmark.py','review_train.py',
                                  'package_release.py','requirements.txt','requirements-export.txt']]
    files += [assets/name for name in ['deim_m.onnx','rtmpose_m26.onnx','model_manifest.json',
                                      'sources.json','export_check.json','DEIMV2_LICENSE.md','RTMPOSE_LICENSE']]
    files += [ROOT/'outputs'/name for name in ['person_scores.csv','scorer.json','validation.json',
                                              'validation_oof.csv','features.csv','features.meta.json',
                                              'benchmark.json','release_checks.json']]
    archive=ROOT.parent/'person_team_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:
            z.write(path,path.relative_to(ROOT.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
        for name in ['deim_m.onnx','rtmpose_m26.onnx']:
            import hashlib
            with z.open('person_pipeline/assets/'+name) as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==models[name]['sha256']
    print(json.dumps({'archive':archive.name,'files':len(files),'size_mib':archive.stat().st_size/2**20,
                      'sha256':digest(archive)},indent=2),flush=True)


if __name__=='__main__':
    main()
