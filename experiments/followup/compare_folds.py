"""Compare paired AP by resampling complete frames within each held-out day."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'outputs' / 'followup_280'
DAYS = ('20261025', '20261026')


def compare():
    records = []
    for day in DAYS:
        ref = pd.read_csv(OUT / f'control_280_{day}.csv')
        groups = [group.index.to_numpy() for _, group in ref.groupby('frame_id')]
        rng = np.random.default_rng(185)
        draws = [np.concatenate([groups[j] for j in ids])
                 for ids in rng.integers(len(groups), size=(500, len(groups)))]
        truth = ref.error.to_numpy()
        draws = [index for index in draws if np.unique(truth[index]).size == 2]
        base_aps = np.array([average_precision_score(truth[i], ref.score.to_numpy()[i]) for i in draws])
        for path in sorted(OUT.glob(f'*_{day}.csv')):
            name = path.stem.removesuffix(f'_{day}')
            if name in ('control_280', 'annotation_style'):
                continue
            data = pd.read_csv(path)
            assert data.id.equals(ref.id) and data.error.equals(ref.error)
            values = data.score.to_numpy()
            deltas = np.array([average_precision_score(truth[i], values[i]) for i in draws]) - base_aps
            lower, upper = np.quantile(deltas, [.025, .975])
            delta = average_precision_score(truth, values) - average_precision_score(truth, ref.score)
            records.append(dict(day=day, variant=name, delta_ap=delta, lower95=lower, upper95=upper,
                                heldout_frames=len(groups), bootstrap_draws=len(draws)))
    result = pd.DataFrame(records)
    result.to_csv(OUT / 'paired_frame_bootstrap.csv', index=False)
    print(result.round(6).to_string(index=False))


if __name__ == '__main__':
    compare()
