# PathSnap Testing Summary (Person 4)

Branch: `qa/testing-integration` (based on `origin/main` @ `489ab2d`). Only new files were added; no existing source, model, feature, evaluation, optimisation or HMM file was changed. No evaluator-test data or ZIP was accessed. All data below is **synthetic**, generated in memory or in pytest temp folders.

## 1. What was added

| File | Purpose |
|---|---|
| `tests/qa/conftest.py` | Synthetic 5x5 grid network, seeded trip generator, trip-file writer, labelled feature matrix (targets come from the generator, never from `.route`) |
| `tests/qa/test_pipeline_integration.py` | Raw files -> candidates -> features -> `run_phase3` artifacts; invalid trip IDs, missing/empty files, empty candidate sets, disconnected route segments, split hygiene |
| `tests/qa/test_model_contract.py` | Same contract run against all five models: `predict_proba` shape/dtype/finiteness/range, class and label handling, NaN/inf, row order, chunking, pickle round trip, leakage guard |
| `tests/qa/test_reproducibility.py` | Fixed-seed determinism of data generation, features, all five models, RF thread-count independence, byte-identical `run_phase3` CSVs |
| `tests/qa/test_api.py` | 15 named placeholder cases, **skipped**: no API branch exists yet |
| `tests/qa/test_benchmark_smoke.py` | Checks the benchmark script runs; asserts nothing about speed |
| `benchmarks/benchmark_inference.py` | Latency and memory benchmark (isolated subprocess per model) |
| `reports/qa/benchmark_results.csv`, `benchmark_environment.json` | Output of one benchmark run (synthetic data, this sandbox) |

## 2. Test results (measured)

Full suite, one run: **217 passed, 15 skipped, 6 xfailed** in 39 s (62 pre-existing tests + 155 new passing tests).

- **15 skipped** are the API placeholders. They are *not* passes: no API tests have run.
- **6 xfailed** are expected failures documenting real gaps (section 4).
- Line coverage of `src/` rose from **69% to 81%** (`pipeline.py` 0% -> 82%, `evaluation.py` 0% -> 75%, `io.py` 49% -> 88%, `mlp.py` and `random_forest.py` 100%). `gps_eda.py` (59%) and `geo.py` (76%) were not targeted.

## 3. Coverage of the requested items

1. **Data -> candidate -> feature integration:** done (schema, IDs preserved, reverse arcs distinct, true arc present for every observation, features independent of the `.route` file).
2. **Five model interfaces:** done, identical tests parametrised over all five.
3. **`predict_proba` shape, finiteness, classes:** done, including label types (bool, Series, Int64, float), non-binary and single-class labels.
4. **API tests:** **not done**. No API code or branch was available. Stubs only.
5. **Invalid trip IDs, missing files, empty candidate sets, disconnected routes:** done (also empty files, negative/fractional route IDs, out-of-range arc IDs).
6. **Performance benchmark scripts:** done and run once (section 5).
7. **Reproducibility with fixed seeds:** done.
8. **This report.**

Not covered: HMM/Viterbi (no code in the repo), determinism of `src.model_evaluation.evaluate_models`, concurrency, any real-data profiling.

## 4. Findings (behaviour observed, nothing fixed)

1. **Leakage guard is inconsistent.** Logistic Regression, Decision Tree and KNN reject columns named `route…`/`label…`; **Random Forest and MLP accept them** (xfail).
2. **Random Forest and MLP accept non-binary labels** (e.g. {0, 2}) and **single-class training labels** (xfail). After single-class training, `predict_proba_model` raises `ValueError` (class 1 missing). The other three models reject both at training time.
3. **Infinite values:** only KNN converts them to NaN; the other four raise `ValueError`.
4. **Zero-row input raises `ValueError` in all five models.** Observations with no candidates are real data (see `candidate_count == 0` tests), so any matcher or API must skip them before scoring.
5. **Passing a NumPy array to a model trained on a DataFrame** works but emits a sklearn feature-names `UserWarning` (seen for four models). Serving code should pass DataFrames with `FEATURE_COLUMNS`.
6. **Pipeline behaviour confirmed:** a missing nodes/arcs/route file or a directory whose name does not match its file prefix is logged as a processing issue, not a crash; student-test route files are never opened; a directory named like the held-out split is not processed; outputs are byte-identical across repeated runs.

## 5. Inference benchmark (one run, synthetic data, **1 CPU / ~4 GB sandbox**)

Models trained on 20,000 synthetic rows with default hyperparameters, 17 features, `predict_proba_model`, median of 20 repeats (3-5 for 100k rows), warm-up excluded.

| Model | 8 rows (one observation) | 800 rows (~one trip) | 100,000 rows |
|---|---:|---:|---:|
| Logistic Regression | 0.83 ms | 1.3 ms | 17 ms |
| Decision Tree | 0.91 ms | 1.7 ms | 18 ms |
| Random Forest | 11.5 ms | 41 ms | 1.95 s |
| KNN | 2.7 ms | 53 ms | 6.57 s |
| MLP | 1.0 ms | 1.5 ms | 69 ms |

| Model | Train time | Pickle size | Model load time | RSS growth on load | Peak RSS while inferring* |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.08 s | 2.5 KB | 0.2 ms | ~0 MB | 206 MB |
| Decision Tree | 0.23 s | 0.08 MB | 0.3 ms | 0.2 MB | 202 MB |
| Random Forest | 8.9 s | 48.1 MB | 52 ms | 82 MB | 300 MB |
| KNN | 0.04 s | 2.9 MB | 1.2 ms | ~0 MB | 220 MB |
| MLP | 1.7 s | 0.09 MB | 0.6 ms | 0.1 MB | 282 MB |

\*Peak RSS includes roughly 180-200 MB of Python/pandas/scikit-learn baseline, so compare models by the differences, not the absolute values.

Caveats: synthetic data (tree depth and KNN neighbour search depend on the data); only 20,000 training rows, whereas the Phase 5 report used about 517,000 per fold, so Random Forest size and KNN latency on the real training set will be **larger** than shown here (KNN stores all training rows); this sandbox has a single CPU, so Random Forest's `n_jobs=-1` gets no parallel speed-up; and a single run on shared hardware carries timing noise. These are not model-quality results. Rerun on the team's hardware with `python -m benchmarks.benchmark_inference` (add `--train-rows 500000` if the real-size comparison is wanted, expect a long Random Forest/MLP training time).

## 6. How to run

```bash
pip install -r requirements.txt pytest pytest-cov psutil
python -m pytest tests/qa -q                       # new QA tests only
python -m pytest -q --cov=src                      # everything
python -m benchmarks.benchmark_inference           # writes reports/qa/benchmark_*.{csv,json}
```

## 7. Open items

- API tests: need the API branch, framework and request/response schema.
- HMM/Viterbi tests: need Person 2's decoder and its transition-model definition.
- Decide whether the Random Forest and MLP owners add the same input guard as the other three models (I did not touch them).
- Optional: pinned dependency file and a CI workflow (not added without approval).
- Real-data profiling and `evaluate_models` determinism: need someone with the training split locally.
