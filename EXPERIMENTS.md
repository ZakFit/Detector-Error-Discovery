# Detector audit experiments

Public results confirmed during the hackathon:

| Submission | Public AP |
|---|---:|
| submission_yolo_pose.csv | 0.83915 |
| submission_crop_person_0842.csv | 0.86558 |
| submission_multiview_duplicates.csv | 0.88899 |
| submission_refinement_person_geometry_half.csv | **0.90734** |

The best confirmed submission uses multi-view crop detection and duplicate scoring
for small objects. For people, it mixes the previous ranking with geometry from
a spatial head trained on frozen DINOv2 features.

The 280×280 candidate, submission_person_refiner_280.csv, mixes the 0.90734
ranking with predictions from three high-resolution refiners. Its day-held-out
AP is 0.94433 / 0.89410 versus 0.93440 / 0.88099 for the confirmed submission.
Its Public score is unknown. Both frame-bootstrap intervals for the gain include zero.

Aggregate validation tables are in experiments/results. Validation holds out
one annotated day at a time. Experiments include crop YOLO, scene matching,
pseudo-labels, masked feature adaptation, occlusion augmentation, semantic
verification, and pairwise box ranking. None of the latter additions produced
a consistent improvement over 0.90734 in the configurations tested.

With the competition data extracted to data/raw and the upstream caches
available, run audit_highres_refiner.py for validation and
audit_highres_refiner.py --test coarse to train on both days. Run
audit_export_highres.py to export the candidate. The backbone is timm
vit_large_patch14_dinov2.lvd142m. GPU scripts use CUDA and the existing
PyTorch, timm, OpenCV, pandas, scipy, scikit-learn and Ultralytics environment.
The scripts expect the chat workspace's outputs directory two levels above
the repo. For a standalone checkout, populate that directory with the
submission CSVs used as inputs. Raw images, weights, and feature caches are
local artifacts.
