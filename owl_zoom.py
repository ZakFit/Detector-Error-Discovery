"""Зум-проверка OWLv2: окно вокруг каждой детекции (руки, чашки, подносы) увеличивается и
проверяется текстовым запросом. Боксы OWLv2 переводятся обратно в пиксели кадра 1280x720.

Веса: google/owlv2-base-patch16-ensemble, https://huggingface.co/google/owlv2-base-patch16-ensemble
Результат: data/owl_zoom.csv — все боксы OWLv2 в окне каждой детекции (id детекции, класс запроса, score, xyxy).
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from features import BOX, D, load

ROOT = Path(__file__).parent
MODEL = 'google/owlv2-base-patch16-ensemble'
QUERIES = {'nude_hands': ['a human hand'], 'cup': ['a cup', 'a glass'], 'tray': ['a tray'], 'person': ['a person']}
# во сколько раз окно больше бокса и его минимальная сторона; людей зумируем слабее — они крупные
SCALE = {'person': 1.6}
MIN_SIDE = {'person': 320}


def window(p):
    w, h = p.x2 - p.x1, p.y2 - p.y1
    side = min(max(SCALE.get(p.class_name, 4.0) * max(w, h), MIN_SIDE.get(p.class_name, 160)), 720)
    cx, cy = (p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2
    x1, y1 = min(max(cx - side / 2, 0), 1280 - side), min(max(cy - side / 2, 0), 720 - side)
    return max(x1, 0), max(y1, 0), side


def main(classes=('nude_hands', 'cup', 'tray'), parts=('test', 'train'), name='owl_zoom'):
    torch.set_num_threads(18)
    processor = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).eval()
    dets = pd.concat([load(p)[0].assign(part=p) for p in parts])
    dets = dets[dets.class_name.isin(classes)]
    # приоритет: руки, затем остальные; внутри — в порядке parts
    dets = dets.assign(prio=(dets.class_name != 'nude_hands') * 2 + (dets.part != parts[0])).sort_values('prio', kind='stable')
    out = ROOT / 'data' / f'{name}.csv'
    # возобновление: пропускаем уже посчитанные детекции (у детекции без боксов OWL строк нет — посчитается заново)
    rows = pd.read_csv(out).to_dict('records') if out.exists() else []
    dets = dets[~dets.id.isin({r['id'] for r in rows})]
    for i, (_, p) in enumerate(dets.iterrows()):
        x0, y0, side = window(p)
        img = Image.open(D / f'{p.frame_id}.jpg').convert('RGB').crop((x0, y0, x0 + side, y0 + side))
        q = QUERIES[p.class_name]
        inputs = processor(text=[q], images=img, return_tensors='pt')
        with torch.no_grad():
            o = model(**inputs)
        res = processor.image_processor.post_process_object_detection(
            o, threshold=0.02, target_sizes=torch.tensor([[side, side]]))[0]
        for (a1, b1, a2, b2), s, k in zip(res['boxes'].tolist(), res['scores'].tolist(), res['labels'].tolist()):
            rows.append({'id': p.id, 'query': q[k], 'score': s,
                         'x1': a1 + x0, 'y1': b1 + y0, 'x2': a2 + x0, 'y2': b2 + y0})
        if i % 25 == 0:
            print(i, len(dets), flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, len(rows))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'person':
        main(classes=('person',), parts=tuple(sys.argv[2:]) or ('train', 'test'), name='owl_zoom_person')
    elif len(sys.argv) > 2 and sys.argv[1] == 'scale':
        # дополнительный масштаб окна для рук: owl_zoom.py scale <множитель> [части...]
        factor = float(sys.argv[2])
        SCALE['nude_hands'], MIN_SIDE['nude_hands'] = factor, 40 * factor
        main(classes=('nude_hands',), parts=tuple(sys.argv[3:]) or ('train', 'test'), name=f'owl_zoom_s{factor:g}')
    else:
        main()
