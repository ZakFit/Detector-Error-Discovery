"""Rebuild the two final CSVs from the reference scores and cached scoring inputs."""

import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal
from pathlib import Path

from verify_results import ROOT, load_submission


REFERENCE = 'submission_v2_visual_geometry_high_upside.csv'
LOCAL = 'submission_v2_final_local_geometry.csv'
EXTENDED = 'submission_v2_final_extended_geometry.csv'


def blend(base, adjustment, weight):
    return base + weight * (adjustment - base)


def render(scores):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\r\n')
    writer.writerow(['id', 'score'])
    writer.writerows((key, f'{value:.12f}') for key, value in scores.items())
    return stream.getvalue().encode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('rebuilt'))
    args = parser.parse_args()

    manifest = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
    entries = {Path(entry['file']).name: entry for entry in manifest}
    reference = load_submission(entries[REFERENCE])
    inputs = json.loads((ROOT / 'final_score_inputs.json').read_text(encoding='utf-8'))

    local = reference.copy()
    for row in inputs['local']:
        if row['id'] not in local:
            raise ValueError(f'Unknown ID {row["id"]}')
        local[row['id']] = round(blend(Decimal(row['base_score']), Decimal(row['risk']), Decimal(row['weight'])), 12)

    extended = local.copy()
    for row in inputs['extended']:
        key = row['id']
        extended[key] = blend(Decimal(str(local[key])), Decimal(row['adjusted_score']), Decimal(row['weight']))

    outputs = {LOCAL: render(local), EXTENDED: render(extended)}
    for name, data in outputs.items():
        if hashlib.sha256(data).hexdigest() != entries[name]['sha256']:
            raise ValueError(f'Rebuilt file differs from archived {name}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = args.output_dir / name
        path.write_bytes(data)
        print(path)


if __name__ == '__main__':
    main()
