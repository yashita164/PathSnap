# Phase 6: Bayesian Optimization Plan

## Baseline selection

Select **Logistic Regression** for optimization. In the saved five-fold grouped
baseline, it achieved the best observation-level Top-1 accuracy:
`0.8404 ± 0.0420`, compared with `0.8284 ± 0.0605` for Random Forest. Random
Forest had lower log loss (`0.1828 ± 0.0631` versus `0.1897 ± 0.0428`) and
higher ROC-AUC and precision, but Top-1 accuracy is the primary PathSnap
objective. Logistic Regression also had higher candidate-level recall and F1
and lower training and inference times.

## Data and objective

- Use only `data/processed/train_candidate_features.csv` and
  `data/processed/train_observations.csv`.
- The baseline source contains 80 training trips; 79 contribute reliable
  labelled candidates. Use the same approved 17 `FEATURE_COLUMNS`, binary
  `is_match` target, and reliable-label filtering as Phase 5. The baseline
  evaluated 646,978 labelled candidate rows (125,938 positive).
- Keep trip, observation, candidate, reference-label, and label-status fields
  outside `X`. Do not use student-test or evaluator data.
- **Optuna objective:** maximize mean observation-level Top-1 accuracy.
  Evaluate Top-1 only for confidently labelled observations with an available
  correct candidate; resolve probability ties by the smallest
  `candidate_arc_id`. Record log loss as a secondary probability-quality
  metric and candidate-level metrics as supporting results.

## Proposed Logistic Regression search space

- Search `C` logarithmically over `[1e-4, 1e3]` (baseline `C=1.0`).
- Keep `solver="lbfgs"`, `max_iter=1000`, `random_state=42`, the L2 penalty
  default, median imputation, and `StandardScaler` fixed. `C` is the primary
  exposed regularization control; only increase the iteration cap if
  convergence warnings are observed, not as a performance-tuning objective.
- Do not search features or change feature definitions.

## Validation strategy

Use nested, trip-aware validation for an unbiased tuned-model estimate. Keep
the initial budget at exactly 30 total trials:

1. Split the labelled training trips into five outer folds with
   `GroupKFold`, grouping by `trip_id`.
2. Within each outer training partition, run Optuna trials scored by a
   four-fold `GroupKFold` objective, also grouped by `trip_id`. Allocate six
   trials to each outer fold.
3. Refit the selected configuration on that outer training partition and score
   it once on the held-out outer trips. Keep all imputation and scaling inside
   the model pipeline and fit them only on each fold's training records.
4. Use the median of the five outer-fold best `C` values as the final
   all-training-data configuration; do not run an additional search beyond the
   30-trial budget. Do not select parameters using student-test results.

Report the five outer-fold metrics and compare them with the saved Phase 5
baseline. Use lower log loss as a tie-breaker when Top-1 results are tied.

## Expected outputs and dependency

- `reports/optimization/optuna_trials.csv` — trial parameters, objective
  values, statuses, and inner-fold scores.
- `reports/optimization/best_parameters.json` — final selected Logistic Regression
  configuration.
- `reports/optimization/optimization_summary.md` — search configuration, validation
  results, and baseline comparison.
- `reports/figures/phase6_optimization_history.png` — per-outer-fold TPE trial
  history.

Optuna is **not installed** in the current Python environment and is not listed
in `requirements.txt`. Add it as a project dependency and install it only when
Phase 6 implementation is authorized.

**Status:** planning only. No optimization was started, and no model
implementation, feature definition, or dataset was modified.
