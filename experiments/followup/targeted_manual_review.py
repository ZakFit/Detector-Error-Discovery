"""Prepare blind panels for detections with high impact on the submitted ranking."""
from PIL import Image
import numpy as np
import pandas as pd

from precise_review import ROOT, panel

OUTPUTS = ROOT / 'outputs'
AUDIT = OUTPUTS / 'manual_bbox_review'
OUT = OUTPUTS / 'followup_280'


def select():
    audit = pd.read_csv(AUDIT / 'detection_audit.csv')
    score = pd.read_csv(OUTPUTS / 'submission_manual_review_full.csv')
    audit = audit.merge(score, on='id', validate='one_to_one')
    score_order = np.sort(audit.score.to_numpy())
    manual = .15 * audit.manual_error_nominal + .85 * audit.boundary_failure_fraction
    # Raise priority when a meaningful change in risk would move many other detections.
    hypothetical = .5 * audit.score + .5 * manual
    current_rank = np.searchsorted(score_order, audit.score)
    proposed_rank = np.searchsorted(score_order, hypothetical)
    audit['impact'] = np.abs(proposed_rank-current_rank) / (1 + .7*current_rank/len(audit))
    selected = (audit[audit.review_label.eq('uncertain')]
                .sort_values('impact', ascending=False).head(30).sort_values('review_number'))
    columns = ['review_number', 'id', 'frame_id', 'class_name', 'score', 'quality', 'manual_error_nominal', 'impact']
    selected[columns].to_csv(OUT / 'targeted_cases.csv', index=False)
    for offset in range(0, len(selected), 6):
        sheet = Image.new('RGB', (1320, 920), '#202126')
        for j, (_, row) in enumerate(selected.iloc[offset:offset+6].iterrows()):
            crop, _ = panel(row)
            sheet.paste(crop, (j % 3 * 440, j // 3 * 460))
        sheet.save(OUT / f'targeted_{offset//6:02}.jpg', quality=93)


if __name__ == '__main__':
    select()
