"""Check the score files and compare their rankings with the reference."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_submission(entry):
    path = ROOT / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'Checksum mismatch in {path.name}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'score']:
            raise ValueError(f'Unexpected columns in {path.name}')
        rows = {row['id']: float(row['score']) for row in reader}
        if reader.line_num != 363 or len(rows) != 362:
            raise ValueError(f'Expected 362 unique rows in {path.name}')
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in rows.values()):
        raise ValueError(f'Invalid score in {path.name}')
    return rows


def top(scores, count):
    return set(sorted(scores, key=scores.__getitem__, reverse=True)[:count])


def main():
    entries = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
    reference_manifest = next(entry for entry in entries if entry.get('reference'))
    reference = load_submission(reference_manifest)
    print('file,reported_public_ap,changed_scores,top50_overlap,top100_overlap')
    for entry in entries:
        scores = load_submission(entry)
        if scores.keys() != reference.keys():
            raise ValueError(f'Different IDs in {entry["file"]}')
        changed = sum(abs(scores[key] - reference[key]) > 1e-10 for key in reference)
        top50 = len(top(reference, 50) & top(scores, 50))
        top100 = len(top(reference, 100) & top(scores, 100))
        public = entry['reported_public_ap']
        print(f'{entry["file"]},{public if public is not None else ""},{changed},{top50},{top100}')


if __name__ == '__main__':
    main()
