# Follow-up to the 280 px refiner

The 280 px model scored 0.91063 Public AP. `submission_manual_review_full.csv`, which blends that model with the visual audit, scored 0.93578 Public AP. The cached 280 control reproduces day-held-out AP of 0.94433015 and 0.89410075.

`audit_280_followup.py` runs the aggregation, five-coordinate-feature, lower-context, targeted-synthetic and multi-box-head ablations. Run `aggregation` first to regenerate the reference predictions. All 280 experiments use seeds 117, 312 and 614. Only the person ranking is changed, with the original 50% rank mixing. The hard-synthetic experiment is separate from lower context. Each held-out split excludes real and synthetic samples from its validation day.

The aggregation control computes risk after averaging offsets and existence logits. `mean_risk` averages `1 − P(exists) × IoU` across seeds. `mean_threshold_risk` replaces IoU with the indicator `IoU ≥ 0.60`. Geometry features are four coordinates relative to the realized crop and its width-to-height ratio. The lower branch adds an embedding of a second view translated down by half a box height when the image boundary permits. Hard examples truncate one side of an official box. The three-hypothesis head predicts weighted alternative boxes and an existence score.

`eval_followup_scores.py` covers inter-class calibration, a prototype duplicate selector using pairwise features aggregated over competing boxes, and a pilot test of manual edge offsets. The pilot fits a regularized class offset on existing blind train annotations with the other day held out. `targeted_manual_review.py` prepares blind panels for 30 high-impact cases. Its decisions are stored in `manual_annotations/targeted_decisions.csv`. `export_followup_submissions.py` verifies the baseline reconstruction and the order of all 362 IDs. It exports conservative manual overrides and blends model variants into the 0.93578 file only when the model variant improved both validation days.

The scripts follow the original workspace layout. They read the competition data and original refiner caches from `data/raw` and `data/gpu_runs/highres_refiner`, and the visual audit and baseline CSVs from `../../outputs`. The local audit archive includes images and reproduction inputs. The repository stores the tabular annotations, scripts and results. The annotated frames, original dataset, large feature caches and model weights remain local.

The manual boxes are a partial reconstruction for the scored detections. Uncertain decisions are recorded explicitly. The visual decisions and changes derived from them have no independent test labels, so their Public AP is unknown until a submission is evaluated.

## Results

| Variant | 2026-10-25 AP | 2026-10-26 AP | Mean |
| --- | ---: | ---: | ---: |
| 280 control | 0.94433 | 0.89410 | 0.91922 |
| Mean seed risk | 0.94433 | 0.88895 | 0.91664 |
| Mean seed threshold risk | 0.94088 | 0.87838 | 0.90963 |
| Geometry features | 0.92977 | 0.88771 | 0.90874 |
| Lower context | 0.93087 | 0.88455 | 0.90771 |
| Truncated-box synthesis | 0.94027 | 0.88941 | 0.91484 |
| Three box hypotheses | 0.94340 | 0.88735 | 0.91537 |
| Class calibration | 0.93207 | 0.89383 | 0.91295 |
| Duplicate selector | 0.94739 | 0.89472 | 0.92106 |

The added context uses a 140 px view while the primary view stays at 280 px. The truncated-box experiment adds 736 synthetic candidates across both folds. They are independent box perturbations. The duplicate selector addresses competing detections separately. The duplicate selector has only 4 to 11 labeled training pairs per fold and the 95% paired frame bootstrap intervals include zero. The learned annotation adjustment improved agreement from 6/9 to 8/9 on the first day and reduced it from 10/15 to 9/15 on the second day. Its fit used 5 to 10 matched manual and official boxes per split. A larger blind audit would be needed to judge the adjustment reliably. The 57 changes it produced on test were excluded from the candidates.

`submission_manual_targeted_plus_duplicate.csv` is the narrowest candidate that includes both visual checks: four detections were downgraded as errors after reviewing 30 ambiguous cases, and one of two boxes around the same hand received a higher error score. `submission_manual_group_constraints.csv` adds seven more local changes where the best visual IoU and lowest model error agree on the winner. `submission_manual_plus_duplicate_pair.csv` blends the prototype selector. All three are compared with the Public 0.93578 file in `candidates.csv`.
