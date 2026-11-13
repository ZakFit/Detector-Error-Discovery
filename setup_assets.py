"""Download public model artifacts into this package, recording provenance."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'


def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'person-audit-hackathon/1.0'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def download(url, path):
    if path.exists():
        print('Cached:', path.name, flush=True)
        return
    print('Downloading:', url, flush=True)
    temporary = path.with_suffix(path.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'person-audit-hackathon/1.0'})
    with urllib.request.urlopen(request, timeout=180) as source, temporary.open('wb') as target:
        shutil.copyfileobj(source, target)
    temporary.replace(path)
    print('Saved:', path.name, path.stat().st_size, 'bytes', flush=True)


def extract(archive, directory):
    directory.mkdir(parents=True, exist_ok=True)
    root = directory.resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            destination = (root / item.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError('Unsafe archive path: ' + item.filename)
        z.extractall(root)


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    provenance = ASSETS / 'sources.json'
    if provenance.exists():
        data = json.loads(provenance.read_text(encoding='utf-8'))
    else:
        repo = get_json('https://api.github.com/repos/Intellindust-AI-Lab/DEIMv2/commits/main')
        hub = get_json('https://huggingface.co/api/models/Intellindust/DEIMv2_DINOv3_M_COCO')
        data = {'deim_code_commit': repo['sha'], 'deim_weights_revision': hub['sha']}
        provenance.write_text(json.dumps(data, indent=2), encoding='utf-8')
    code_url = 'https://codeload.github.com/Intellindust-AI-Lab/DEIMv2/zip/' + data['deim_code_commit']
    weights_base = 'https://huggingface.co/Intellindust/DEIMv2_DINOv3_M_COCO/resolve/' + data['deim_weights_revision']
    pose_url = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
                'rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip')
    files = [
        ('deim_code.zip', code_url),
        ('deim_m.safetensors', weights_base + '/model.safetensors'),
        ('deim_config.json', weights_base + '/config.json'),
        ('rtmpose_m26.zip', pose_url),
        ('RTMPOSE_LICENSE', 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/LICENSE'),
    ]
    data['artifacts'] = []
    for name, url in files:
        path = ASSETS / name
        download(url, path)
        data['artifacts'].append({'file': name, 'url': url, 'sha256': sha256(path), 'bytes': path.stat().st_size})
        provenance.write_text(json.dumps(data, indent=2), encoding='utf-8')
    extract(ASSETS / 'deim_code.zip', ASSETS / 'deim_source')
    extract(ASSETS / 'rtmpose_m26.zip', ASSETS / 'pose_source')
    print('Artifacts ready.', flush=True)


if __name__ == '__main__':
    main()
