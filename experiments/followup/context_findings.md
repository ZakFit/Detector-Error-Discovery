# Context and validation audit

Public AP is 0.93578 for the original manual submission, the duplicate-pair blend and the five-edit manual candidate. The new candidates start from the original manual submission.

The 280 control covers 198 unique detections in 35 frames on two held-out days. All 472 other train rows are `table`, a class excluded from the test. Recomputing one-to-one matching at IoU ≥ 0.60 with official boxes reproduced every one of the 198 labels. The two days contribute 68 and 130 detections and 34 and 75 positive errors. No listed neighbor crosses day boundaries. The final 280 model's source training mask excludes the held-out day for both real and synthetic rows. These two dates were reused during model selection. This is a retrospective audit, not a rerun of every upstream checkpoint under nested validation.

| Method | Day 25 AP | Day 26 AP | Pooled OOF AP |
| --- | ---: | ---: | ---: |
| Control 280 | .94433 | .89410 | .90653 |
| Neighbor-frame rule | .94908 | .88592 | .90372 |
| Relaxed background rule | .94665 | .89558 | .90830 |
| Independent matching, same reference views | .94825 | .89599 | .90998 |
| Joint matching | .94825 | .89761 | .91121 |

The automated neighbor rule changed 21 train scores, including 13 actual errors that would have been promoted as correct. Review of 20 selected test episodes helped clarify objects and occlusions but did not justify a stand-alone temporal override. Strict background matching had no coverage. The exploratory relaxed rule affected seven train rows, six actual errors, and twelve test rows. Visual inspection confirmed six specific false trays: plates placed directly on the table, five in the same episode. The remaining visually ambiguous background hits were excluded.

Joint matching uses a maximum-cardinality assignment before IoU tie-breaking. Its added signal over independent matching changed ten test rows, some disagreeing with the visual audit. Three ambiguous frames were reviewed in full and the selected duplicate corrections are a cup and a hand, rows 137 and 173. The manual candidates raise the error scores of six visually verified false trays, with the larger candidate also softly raising those two duplicate scores. The adjustments are heuristic. Public AP of the two candidates is unknown.

The original .91922 is the mean of the two separate daily APs. Pooled AP evaluates the joint rank and is closer in form to leaderboard scoring, but separately trained folds can have distinct score distributions. Report pooled, daily and per-class diagnostics together. On a future independent set, hold out whole days and use the same scoring pipeline, keeping tuning separate. Bootstrap groups of related frames rather than individual detections. With 500 resamples of time episodes split at gaps over five minutes, the joint-matching delta intervals were approximately [-.00002, .01259] and [-.00251, .01094]. The small test of background threshold 24 was exploratory and both day intervals included zero. AP against our own test labels is a consistency diagnostic only.
