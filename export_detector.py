"""Export the public DEIMv2-M safetensors checkpoint on CPU, batch one."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'


def build_model():
    import torch
    from safetensors.torch import load_file
    source = next((ASSETS / 'deim_source').glob('DEIMv2-*'))
    sys.path.insert(0, str(source))
    from engine.backbone import DINOv3STAs
    from engine.deim import HybridEncoder, DEIMTransformer, PostProcessor
    config = json.loads((ASSETS / 'deim_config.json').read_text())

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DINOv3STAs(**config['DINOv3STAs'])
            self.encoder = HybridEncoder(**config['HybridEncoder'])
            self.decoder = DEIMTransformer(**config['DEIMTransformer'])
            self.postprocessor = PostProcessor(**config['PostProcessor'])

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.decoder(self.encoder(self.backbone(images))), orig_target_sizes)

    model = Model()
    state = load_file(str(ASSETS / 'deim_m.safetensors'))
    # HF safetensors stores tied tensors once; restore their two state-dict aliases.
    for name in ['up','reg_scale']:
        assert getattr(model.decoder,name) is getattr(model.decoder.decoder,name)
        state['decoder.'+name] = state['decoder.decoder.'+name]
    model.load_state_dict(state, strict=True)
    model.eval()
    for part in list(model.modules()):
        if hasattr(part, 'convert_to_deploy'):
            part.convert_to_deploy()
    model.postprocessor.deploy()
    return model


def main():
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from PIL import Image
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026)
    started = time.perf_counter()
    model = build_model()
    image = Image.open(next((ROOT.parent / 'data' / 'imgs').glob('*20261025*.jpg'))).convert('RGB')
    original_size = torch.tensor([[image.width, image.height]], dtype=torch.int64)
    array = np.asarray(image.resize((640, 640), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    array = (array - np.array([.485,.456,.406],np.float32)) / np.array([.229,.224,.225],np.float32)
    tensor = torch.from_numpy(array.transpose(2,0,1)[None].copy())
    with torch.inference_mode():
        expected = [x.numpy() for x in model(tensor, original_size)]
    print('Loaded weights strictly. Forward complete.', time.perf_counter()-started, flush=True)
    destination = ASSETS / 'deim_m.onnx'
    with torch.no_grad():
        torch.onnx.export(model, (tensor, original_size), str(destination),
                          input_names=['images','orig_target_sizes'],
                          output_names=['labels','boxes','scores'], opset_version=17,
                          dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'images':tensor.numpy(),'orig_target_sizes':original_size.numpy()})
    # Top-k ordering can differ only at tiny score ties; check the first 50 confident outputs.
    errors = {name:float(np.max(np.abs(a[:,:50].astype(float)-b[:,:50].astype(float))))
              for name,a,b in zip(['labels','boxes','scores'], expected, actual)}
    if errors['labels'] or errors['boxes'] > .2 or errors['scores'] > .001:
        raise RuntimeError('ONNX verification failed: ' + str(errors))
    report = {'max_abs_difference_first_50':errors,'seconds':time.perf_counter()-started,
              'providers':session.get_providers(),'batch_size':1,'precision':'FP32',
              'torch':torch.__version__, 'onnxruntime':ort.__version__}
    (ASSETS/'export_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
