# Phase 6 Final Tuned Logistic Regression

## Final refit and artifact

- Selected parameters: `{'C': 2.778663410880327, 'max_iter': 1000, 'solver': 'lbfgs', 'random_state': 42, 'impute_strategy': 'median'}`.
- Fit on 646,978 reliable candidate rows from 79 eligible trips (80 source training trips).
- Only the existing training split was used for fitting; preprocessing remains inside the fitted pipeline.
- Training time: 10.324 seconds.
- Artifact: `C:\Users\Yashita Anand\OneDrive\Desktop\Pathsnap\artifacts\model.joblib` (1,846 bytes), saved and loaded successfully with joblib.
- The loaded pipeline preserved class order `[0, 1]` and produced finite, row-aligned probabilities.

## Student-test evaluation

Student-test route files were used only to create route-aligned evaluation labels; they were not used for fitting, feature generation, threshold selection, or model selection.

- Trips found: 10; evaluated: 9; failed: 1.
- Candidate rows evaluated: 54,381; labelled observations: 10,017.

| Metric | Result |
|---|---:|
| accuracy | 0.940310 |
| precision | 0.825038 |
| recall | 0.857671 |
| f1 | 0.841038 |
| roc_auc | 0.973235 |
| log_loss | 0.160370 |
| top1_accuracy | 0.873152 |
| candidate_miss_rate | 0.000499 |
| training_seconds | 10.324073 |
| inference_seconds | 0.130413 |

Trips not evaluated (with reasons):
- `00000000`: FileNotFoundError: Required dataset file was not found: C:\Users\Yashita Anand\OneDrive\Desktop\Pathsnap\data\student_test\00000000\00000000.arcs

## Tuned-versus-untuned context

The comparison CSV keeps three protocols distinct: Phase 5 untuned GroupKFold, Phase 6 nested GroupKFold, and the final tuned model on student-test.

| Evaluation protocol | Configuration | Top-1 accuracy | Log loss | Training (s) | Inference (s) | Candidate miss rate |
|---|---|---:|---:|---:|---:|---:|
| Phase 5 five-fold GroupKFold | C=1.0, max_iter=1000, solver=lbfgs, random_state=42 | 0.840363 ± 0.041984 | 0.189662 ± 0.042768 | 5.781 | 0.377 | n/a |
| Phase 6 five-fold nested GroupKFold | Fold-specific Optuna C; final selected C=2.77866341088 | 0.840235 ± 0.041769 | 0.189267 ± 0.043065 | 4.951 | 0.105 | n/a |
| Final refit; student-test, one evaluation | Tuned final refit, C=2.77866341088 | 0.873152 | 0.160370 | 10.324 | 0.130 | 0.000499 |

- Phase 6 nested tuned Top-1 differed from the Phase 5 untuned baseline by -0.000128; log loss differed by -0.000394 (negative is lower). These are different validation protocols, so the differences do not establish a tuning improvement.

The student-test C=1 baseline was not refit, per the authorized tuned-only scope; therefore there is no matched tuned-versus-untuned student-test comparison and no claim of improvement from cross-protocol results.

Observation-level Top-1 accuracy remains the primary selection metric. Log loss is reported as a secondary probability-sensitive metric because downstream map matching consumes classifier probabilities; lower log loss penalizes confident errors.

## Reproduction

From the repository root, run:

```powershell
python -m src.optimization.finalize_logistic_regression
```

To regenerate only the model artifact without reading student-test data:

```powershell
python -m src.optimization.finalize_logistic_regression --artifact-only
```

API model path: set `PATHSNAP_MODEL_PATH` to `C:\Users\Yashita Anand\OneDrive\Desktop\Pathsnap\artifacts\model.joblib`. The API configuration is not changed by this workflow.
