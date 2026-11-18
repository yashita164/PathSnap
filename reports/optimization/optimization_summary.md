# PathSnap Phase 6 Logistic Regression Optimization

## Search configuration

- Total Optuna trials: 30, allocated across five outer folds.
- Each outer training partition has its own study with a four-fold `GroupKFold` objective; six trials are used per fold for the initial 30-trial run.
- Sampler: seeded `TPESampler`; sampler seed is `base_seed + outer_fold - 1`. Each study uses two startup trials and runs sequentially.
- Objective: maximize mean inner-fold observation-level Top-1 accuracy. `C` is searched log-uniformly in [0.0001, 1000].
- Fixed classifier settings: solver `lbfgs`, max_iter `1000`, random_state `42`; median imputation and scaling remain inside the Logistic Regression pipeline.
- Outer-fold best C values: `[612.0117715317866, 685.1043526785173, 1.2240981404645566, 0.7027697960311682, 2.778663410880327]`.
- Final reported C: **2.7786634**, the median of the five outer-fold best C values. No additional study was run beyond the 30-trial budget.

## Nested outer-fold results

Mean ± standard deviation across the five held-out-trip outer folds:

| Metric | Mean ± standard deviation |
|---|---:|
| accuracy | 0.9260 ± 0.0218 |
| precision | 0.8109 ± 0.0332 |
| recall | 0.8125 ± 0.0540 |
| f1 | 0.8114 ± 0.0413 |
| roc_auc | 0.9649 ± 0.0130 |
| log_loss | 0.1893 ± 0.0431 |
| top1_accuracy | 0.8402 ± 0.0418 |
| training_seconds | 4.9506 ± 1.8489 |
| inference_seconds | 0.1050 ± 0.0384 |

All five outer folds had zero trip overlap between training and validation. Inner-fold preprocessing was fitted only on each inner training partition; outer evaluation preprocessing was fitted only on the corresponding outer training partition.

## Phase 5 baseline comparison

- Phase 5 five-fold GroupKFold Top-1 accuracy: 0.8404 ± 0.0420.
- Phase 5 log loss: 0.1897 ± 0.0428.
- The Phase 5 comparison file was read only and was not overwritten. Its score is a non-nested baseline; the optimized score above is from nested outer folds, so the protocol difference should be considered when comparing them.

## Trial accounting and limitations

- Trial rows recorded: 30; statuses: `{'COMPLETE': 30}`.
- Only the prepared training candidate and observation datasets were used. Ambiguous/unreliable labels are excluded by the shared Phase 5 preparation helper; trip and reference-label columns are metadata, not model features.
- The final C is an aggregate of fold-specific optima. It is not the result of a separate full-data Optuna search because the trial budget was capped at 30.
