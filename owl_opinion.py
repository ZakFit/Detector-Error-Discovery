"""Второе мнение open-vocabulary: OWLv2 по текстовым запросам на целевых кадрах.

Веса: google/owlv2-base-patch16-ensemble, https://huggingface.co/google/owlv2-base-patch16-ensemble
Результат: data/second_opinion_owlv2.csv, боксы в пикселях 1280x720.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from features import D

ROOT = Path(__file__).parent
MODEL = 'google/owlv2-base-patch16-ensemble'
QUERIES = ['a human hand', 'a cup', 'a glass', 'a tray', 'a person', 'a jacket', 'a bag', 'a plate', 'a phone']


def main():
    torch.set_num_threads(18)
    processor = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).eval()
    frames = pd.concat([pd.read_csv(D / f'{p}_context.csv').frame_id for p in ('train', 'test')]).unique()
    out = ROOT / 'data' / 'second_opinion_owlv2.csv'
    rows = []
    for i, fid in enumerate(frames):
        img = Image.open(D / f'{fid}.jpg').convert('RGB')
        inputs = processor(text=[QUERIES], images=img, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        # OWLv2 паддит изображение до квадрата: целевой размер — сторона квадрата
        side = max(img.size)
        res = processor.image_processor.post_process_object_detection(outputs, threshold=0.05,
                                                      target_sizes=torch.tensor([[side, side]]))[0]
        for (x1, y1, x2, y2), s, k in zip(res['boxes'].tolist(), res['scores'].tolist(), res['labels'].tolist()):
            rows.append({'frame_id': fid, 'ext_class': QUERIES[k], 'ext_conf': s, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        if i % 10 == 0:
            print(i, len(frames), flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, len(rows))


if __name__ == '__main__':
    main()
