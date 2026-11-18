# PathSnap Phase 5 Model Evaluation

## Dataset and preparation

- Candidate rows in the prepared training artifact: 1,084,832.
- Candidate rows used for supervised evaluation: 646,978 (125,938 positive, 521,040 negative).
- Candidate rows excluded: 437,854.
- Excluded by label status: `{'ambiguous': 412973, 'no_candidate': 21186, 'infeasible': 2587, 'component_conflict': 1108}`.
- Labelled candidates excluded for invalid targets: 0.
- Labelled candidates excluded because no valid observation-level ground truth was available: 0.
- Source training trips: 80; trips contributing reliable candidate rows: 79; trips with no reliable candidates: 1.
- Candidate-level numeric feature columns (17): point_to_segment_distance_m, distance_to_start_node_m, distance_to_end_node_m, candidate_arc_length_m, candidate_road_bearing_deg, candidate_bearing_available, gps_movement_bearing_deg, gps_heading_available, bearing_difference_deg, heading_consistency, gps_speed_mps, gps_speed_kmh, gps_speed_available, time_delta_s, time_interval_valid, time_gap_over_5s, time_gap_over_60s.
- Missing values in the selected feature matrix before fold-local imputation: 0; infinite values converted to missing: 0.
- `trip_id`, `gps_observation_index`, and `candidate_arc_id` were retained as metadata and excluded from `X`; label/reference columns were not features.
- Only the prepared training candidate and observation artifacts were read. Student-test and evaluator data were not used.

## Validation procedure

- 5-fold `GroupKFold`, grouped by `trip_id`; each fold uses the same candidate rows and held-out trips for all five models. All recorded train/validation trip-overlap checks were zero.
- Mean candidate sample counts per fold: 517,582 training and 129,396 validation.
- Imputation and any model-specific scaling are fit inside each model pipeline on that fold's training rows only; tree-based models are unscaled.
- Candidate-level recall is positive-class classifier recall, not Phase 3 candidate recall.
- For Top-1, only confidently labelled observations with an available correct candidate are counted. Equal-probability ties are resolved by the smallest `candidate_arc_id`; any positive candidate is considered correct.
- Valid ground-truth observations: 126,767; with a correct candidate available: 125,938; without one: 829.
- Observation groups with multiple positive candidate rows: 0.

## Five-fold results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss | Top-1 accuracy | Training time (s) | Inference time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9256 ± 0.0222 | 0.8122 ± 0.0328 | 0.8078 ± 0.0569 | 0.8096 ± 0.0423 | 0.9643 ± 0.0136 | 0.1897 ± 0.0428 | 0.8404 ± 0.0420 | 5.7806 ± 3.9016 | 0.3771 ± 0.2923 |
| Decision Tree | 0.9181 ± 0.0280 | 0.7915 ± 0.0446 | 0.7919 ± 0.0724 | 0.7915 ± 0.0580 | 0.9598 ± 0.0147 | 0.2512 ± 0.0607 | 0.8200 ± 0.0543 | 17.1824 ± 4.7763 | 0.2666 ± 0.0655 |
| Random Forest | 0.9237 ± 0.0294 | 0.8198 ± 0.0511 | 0.7867 ± 0.0729 | 0.8026 ± 0.0612 | 0.9667 ± 0.0185 | 0.1828 ± 0.0631 | 0.8284 ± 0.0605 | 153.4724 ± 119.9792 | 7.9375 ± 4.7504 |
| MLP | 0.9158 ± 0.0325 | 0.7872 ± 0.0513 | 0.7860 ± 0.0817 | 0.7863 ± 0.0666 | 0.9617 ± 0.0198 | 0.2096 ± 0.0794 | 0.8107 ± 0.0664 | 394.8391 ± 175.4606 | 0.5781 ± 0.4835 |
| Distance-weighted KNN | 0.8998 ± 0.0305 | 0.7556 ± 0.0496 | 0.7242 ± 0.0693 | 0.7393 ± 0.0589 | 0.9077 ± 0.0377 | 1.3592 ± 0.6558 | 0.7746 ± 0.0623 | 1.9460 ± 0.5924 | 101.6441 ± 52.6035 |

All metric and timing entries above are mean ± standard deviation across the 5 held-out-trip folds. Fold-wise values and sample counts are saved in `model_fold_results.csv`; aggregate values are in `model_comparison.csv`.

## Best model and trade-offs

**Selected model: Logistic Regression.** Selection follows the stated priority: highest mean observation-level Top-1 accuracy, then lower mean log loss, then candidate precision, recall, F1, and training/inference time. Its measured mean Top-1 accuracy was 0.8404 ± 0.0420; mean log loss was 0.1897 ± 0.0428.

This is a grouped cross-validation comparison on the current training trips, not an external test-set result. The folds share one finite set of trips, so the reported standard deviations describe fold variation and are not independent confidence intervals.

Comparison figures are saved under `figures/`.
