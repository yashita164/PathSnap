# PathSnap Testing and Integration Plan (Person 4)

Status: **plan only. No implementation has started.** Written against `origin/main` at commit `489ab2d`.

## 0. Ground rules followed

- Model implementations, feature definitions, `src/model_evaluation.py` and optimisation code are not modified. Anything I find is reported, not fixed.
- No evaluator-test data or evaluator ZIP was opened. `data/` is gitignored and absent from the repo, so all planned tests use **synthetic fixtures built in `tmp_path`**.
- No fake predictions or metrics. Every number below is either measured by me in a scratch clone (labelled "measured") or taken from the repo's own `reports/` files (labelled "repo report"). Latency budgets in section 7 are *proposals to agree on*, not results.
- New work goes only in new files (`tests/…`, `tests/conftest.py`, optional `pytest.ini`/CI file). No existing source file changes.

## 1. What the repo contains today

| Area | Module | Role |
|---|---|---|
| Loading | `src/io.py` | Headerless whitespace tables (`.track/.nodes/.arcs/.route`); validates numeric, finite, non-negative integer IDs; row position = ID |
| Geometry | `src/geo.py` | Projection, point-to-segment distance, bearings, headings (lon/lat in, metres out) |
| Candidates | `src/candidates.py` | `generate_candidates(track, nodes, arcs, trip_id, search_radius_m=50, max_candidates=8)` -> candidates + per-observation counts + invalid arc IDs |
| Features | `src/features.py` | `extract_candidate_features(...)` -> 3 identifier cols + 17 numeric `FEATURE_COLUMNS`; route-independent; unavailable values = 0 + availability flag |
| Labels | `src/labeling.py` | `align_route_labels(...)`: route used only for training labels; ambiguous/infeasible/conflict rows left unlabelled |
| Phase 3 | `src/pipeline.py` | `run_phase3(data_root, ...)` reads `train/` and `student_test/` only; writes CSVs to `data/processed` |
| Models | `src/models/{logistic_regression,decision_tree,random_forest,knn,mlp}.py` | Each exposes `build_model / train_model / predict_model / predict_proba_model` returning an sklearn `Pipeline` |
| Evaluation | `src/model_evaluation.py` | 5-fold `GroupKFold` by `trip_id`, Top-1 accuracy per observation, writes `reports/` |
| Tests | `tests/` (9 files, 62 tests) | geo, gps_eda, phase3 pieces, each model, model_evaluation |
| API / UI | **none found** | No FastAPI/Flask/Streamlit/app code and no dependency for one in `requirements.txt` |
| HMM / Viterbi | **none found** | No module, test or doc references it |

So sections 3 (API) and 5 (HMM/Viterbi) can only be planned against an *assumed contract*, and I need your decisions listed in section 9.

## 2. Baseline (measured, one run)

`pytest --cov=src` on `main`: **62 passed in 12.6 s, 69% line coverage overall.**

| Module | Coverage | Note |
|---|---:|---|
| `src/pipeline.py` | **0%** | `run_phase3` has no test at all; `tests/test_phase3.py` covers candidates/labels/features only |
| `src/evaluation.py` | **0%** | route-reference candidates and recall summary untested |
| `src/io.py` | 49% | most error branches (missing/empty/unparseable/negative-ID files) untested |
| `src/gps_eda.py` | 59% | plotting/report paths; lower priority |
| `src/geo.py` | 76% | validation branches |
| `src/labeling.py` | 83% | |
| `src/model_evaluation.py` | 89% | `load_training_data` (lines 138-174) untested |
| `src/models/*` | 88-96% | |
| `src/candidates.py`, `src/features.py` | 90% | validation branches |

There is no `pytest.ini`/`pyproject.toml`, no `conftest.py`, no CI workflow, and `requirements.txt` has only lower bounds (`>=`), which affects reproducibility (section 8).

## 3. Model probability interface compatibility (measured findings)

I trained each of the five models on the same 200x3 synthetic matrix in a scratch clone and probed edge cases. This is an interface check only, not a metric.

| Behaviour | LogReg | DecTree | RandForest | KNN | MLP |
|---|---|---|---|---|---|
| Rejects a column named `route…`/`label…` | yes | yes | **no** | yes | **no** |
| `+inf` in predict input | ValueError | ValueError | ValueError | **handled** | ValueError |
| Wrong column count at predict | ValueError | ValueError | ValueError | ValueError | ValueError |
| Single-class `y` at train | ValueError | ValueError | **accepted** | ValueError | **accepted** |
| Zero-row `X` at predict | ValueError | ValueError | ValueError | ValueError | ValueError |
| `predict_proba_model` takes `batch_size` | no | no | no | **yes** | no |

What matters for integration:

1. **Leakage guard is inconsistent.** Random Forest and MLP (MLP is the file from my earlier task) accept a `route`/`label` column; the other three refuse. The shared feature matrix is clean today, but an API layer must not rely on per-model guards. Proposed: a model-agnostic contract test plus an integration-layer column allow-list based on `FEATURE_COLUMNS`. A decision on whether to align the models is for the owners (section 9).
2. **Inf handling differs.** KNN converts inf to NaN; the others raise. The pipeline's features are documented finite, so the API layer should reject or sanitise non-finite values itself.
3. **Empty candidate sets are real.** The repo report shows 21,186 `no_candidate` rows and `extract_candidate_features` returns an empty frame. Every model raises on zero rows, so the matcher must skip empty observations before calling a model.
4. **Single-class training data** is silently accepted by RF/MLP, after which `predict_proba_model` can only fail. Training pipelines should check class balance before fitting.
5. `model_evaluation._predict_validation_batch` already enforces shape `(n,)`, finite values and `[0, 1]`; the same checks should be a shared contract test applied to **all five** models.

Planned contract tests (`tests/integration/test_model_contract.py`, parametrised over the five modules):
- `predict_proba_model` returns float64, shape `(n,)`, finite, within [0, 1].
- `predict_model` returns only {0, 1}, shape `(n,)`.
- Row order and count are preserved; shuffling input rows shuffles outputs identically.
- Same feature matrix (`FEATURE_COLUMNS` order, DataFrame and ndarray) works for all five.
- NaN accepted everywhere; behaviour on inf and on forbidden column names is *recorded* in an expected-behaviour table, so a later change shows up as a deliberate diff.
- Pickle/joblib round-trip yields identical probabilities (needed for API model loading).
- Threshold consistency: `predict_model == (proba >= 0.5)` where the model claims to (KNN does; check for the others and document the difference rather than assert it).

## 4. Unit test coverage plan

Targets are for `src/` excluding `gps_eda.py` plotting. Priorities: P0 blocks integration, P1 should land early, P2 is nice to have.

| Priority | Module | Add |
|---|---|---|
| P0 | `src/pipeline.py` | Synthetic mini dataset in `tmp_path` (`train/` + `student_test/`, 2-3 tiny trips); assert output CSV names, columns (`IDENTIFIER_COLUMNS` + `FEATURE_COLUMNS` + `dataset_split`), `student_test` route files are never opened (make them unreadable or absent), issue log entries for missing/invalid files |
| P0 | `src/io.py` | Missing file, empty file, wrong column count, non-numeric, NaN/inf, negative / fractional / over-int64 IDs, whitespace variants, CRLF, trailing blank lines |
| P0 | `src/evaluation.py` | `build_route_reference_candidates` and `candidate_recall_summary` on hand-computed geometry |
| P1 | `src/candidates.py` | Radius/max-candidate validation, reverse arcs distinct, invalid arc IDs reported, NaN GPS rows, exactly-on-radius boundary, ranking ties |
| P1 | `src/features.py` | Each availability flag, zero-speed, repeated/decreasing timestamps, `minimum_heading_speed_mps` edge, no route input accepted, output finite |
| P1 | `src/model_evaluation.py` | `load_training_data`, `n_splits` > trips error, tie-break rule (smallest `candidate_arc_id`) |
| P2 | `src/geo.py`, `src/labeling.py` | Remaining validation branches |

Coverage gate proposal: report per-module coverage in CI; fail below 85% for `io`, `candidates`, `features`, `pipeline`, `labeling`; no gate on `gps_eda`.

## 5. End-to-end pipeline tests

Synthetic trips are built by a shared fixture generator in `tests/conftest.py` (grid road network, straight/L-shaped route, GPS with seeded Gaussian noise). No real data.

1. **Raw -> features:** `load_trip -> generate_candidates -> extract_candidate_features` on the synthetic trip. Assert identifier columns, 17 feature columns in `FEATURE_COLUMNS` order, all finite, one row per candidate pair.
2. **Features -> model -> probabilities:** train each of the five models on synthetic labelled rows, predict on a held-out synthetic trip, check the contract in section 3 and that the true arc gets the highest probability on a clean low-noise trip (synthetic data only; this is a plumbing check, not a performance claim).
3. **Probabilities -> per-observation decision** (Top-1 with the same tie rule as evaluation).
4. **Leakage test:** run the whole chain with `.route` files deleted for a `student_test`-style trip; output must be identical.
5. **Split hygiene:** `GroupKFold` folds have zero trip overlap; no `evaluator` path is ever touched (monkeypatch `open`/`Path.open` to fail on any path containing `evaluator`).
6. **Artifact schema test:** the columns `model_evaluation.load_training_data` expects match what `run_phase3` writes, so Phase 3 -> Phase 5 cannot drift silently.

## 6. API endpoint tests

No API code exists, so this section is a template that I will bind to the real framework once decided. Assumed contract (to confirm, section 9): a request carries a GPS track and a road network (or a network ID), and the response returns the matched arc per observation plus optional confidence.

Test groups (using the framework's in-process test client; no server, no network):
- **Health/metadata:** health endpoint, model name/version, feature schema.
- **Happy path:** valid synthetic track -> 200, response schema validated, one result per observation, arc IDs exist in the network, confidences in [0, 1].
- **Validation (expect 4xx, never 500):** empty track, single point, missing field, wrong types, lat outside [-90, 90], lon outside [-180, 180], NaN/inf/null, non-monotonic or duplicate timestamps, huge payload, unknown fields, `route` field supplied (must be ignored or rejected, never used).
- **No-candidate observations:** off-network point returns a defined "unmatched" result, not an exception.
- **Model unavailable/corrupt:** clear 5xx/503 with no stack trace or file path in the body.
- **Determinism:** same request twice -> identical response.
- **Concurrency:** N parallel requests return correct, non-interleaved results (shared pipeline objects are not mutated by `predict`).
- **Swap-the-model test:** the same request passes through each of the five models via configuration.

## 7. HMM / Viterbi integration tests

There is no implementation, so these tests define the **interface the decoder must satisfy**:

- **Input contract:** per observation, a 1-D probability vector for its candidate arcs from `predict_proba_model`, aligned to candidate rows, plus the arc topology from `.arcs` (`from_node -> to_node`) for transitions.
- **Hand-built network oracle:** a 5-6 arc network where the correct Viterbi path is derivable by hand; assert exact path and log-score.
- **Numerical safety:** probabilities of exactly 0 or 1 do not produce `-inf`/NaN (clamp or log-space handling); long sequences (thousands of observations) do not underflow.
- **Connectivity:** unreachable transitions are penalised or break the chain, as designed; arcs with reversed direction are separate states.
- **Gaps:** observations with no candidates (section 3, item 3) split or bridge the sequence in a defined way, never crash.
- **Ties:** equal scores resolved deterministically (smallest `candidate_arc_id`, matching evaluation).
- **Consistency check:** with a transition model that is uniform, Viterbi must reduce to per-observation argmax, which equals the Top-1 decision from section 5.
- **Model-agnostic:** same decoder test parametrised over all five models' probability outputs on a synthetic trip.
- **Determinism and length invariance:** identical output on repeat; a sequence of length 1 and 2 behave sensibly.

## 8. Invalid input and missing-data handling

A single table-driven test module, `tests/integration/test_invalid_inputs.py`, with one row per case and the expected exception or defined fallback:

- **Files:** missing/empty/unreadable, wrong delimiter, extra or missing columns, non-numeric, NaN/inf, negative/fractional IDs, arc referencing a non-existent node, empty route (training only).
- **GPS:** coordinates out of range, NaN rows, duplicate points, zero time delta, decreasing timestamps, one-point track, stationary track.
- **Network:** zero-length arcs, self-loops, isolated nodes, duplicate arcs.
- **Features/model:** NaN features (all models), all-NaN column (MLP is covered by an existing test), inf, wrong column count/order, zero rows, single-class labels, string columns.
- **Rule:** user-facing code paths raise a clear `ValueError` (or return a documented "unmatched"), never an unhandled `IndexError`/`KeyError`/numpy warning-turned-NaN.

## 9. Prediction latency and memory profiling

Tooling: `time.perf_counter`, `tracemalloc` (Python allocations), `psutil` RSS (process memory), optional `pytest-benchmark`. Profiling lives in `tests/perf/` behind a `-m perf` marker so normal runs stay fast.

What I will measure per model, on **synthetic** data shaped like the real matrix (17 features, up to 8 candidates per observation):
- Model load time and serialized size (joblib).
- `predict_proba_model` latency for 8 rows (one observation), 800 rows (~one trip), and 100k rows (batch), with warm-up and median/p95 over repeats.
- Peak RSS during load and during a 100k-row predict.
- Training time and peak memory on a fixed synthetic size (informational).

Context from the repo's own report (`model_comparison.csv`, 5-fold means over ~129k validation rows, hardware unspecified): inference was ~0.38 s (LogReg), 0.27 s (Tree), 7.9 s (RF), 0.58 s (MLP) and **~101.6 s (KNN)**. That works out to roughly 3 microseconds per row for LogReg versus ~0.8 ms per row for KNN, and KNN also keeps its whole training set in memory. This is derived from their numbers, not my measurement, and my profiling will confirm or contradict it on common hardware. The expectation is that KNN is the integration risk for any interactive API.

Proposed budgets (to be agreed, not claims): single-observation prediction < 50 ms and a ~800-row trip batch < 1 s for the model chosen for the API; model load < 5 s; fail the perf job on a >25% regression against a stored baseline produced from my own runs. Real-data profiling needs `data/processed` (training split only) and must be run locally by a teammate who has it.

## 10. Reproducibility checks

- **Seeds:** each model's `random_state` defaults to 42. Test: two trainings on identical data give identical probabilities (exact for LogReg/Tree/RF/KNN; `allclose` with tight tolerance for MLP because BLAS threading can perturb float order). RF is run with `n_jobs=-1` and `n_jobs=1` and compared.
- **Pipeline determinism:** `run_phase3` twice on the same synthetic root -> byte-identical CSVs (hash compare); trip directory order is sorted already, so this should hold, and the test locks it in.
- **Evaluation determinism:** `GroupKFold` fold assignment identical across runs; `evaluate_models` on a tiny synthetic frame returns identical metrics (excluding timing columns).
- **Environment:** record `pip freeze` in the test report. `requirements.txt` uses only `>=`, so I recommend a lock/constraints file for CI. Recorded as a risk, not changed by me.
- **Serialization:** save/load round-trip and loading under a different working directory.
- **Order independence:** shuffling candidate rows within a trip does not change the per-observation decision.

## 11. Proposed layout and execution

```
tests/
  conftest.py                    # synthetic trip/network/feature generators (new)
  integration/
    test_model_contract.py       # section 3
    test_pipeline_e2e.py         # section 5
    test_invalid_inputs.py       # section 8
    test_reproducibility.py      # section 10
    test_viterbi_contract.py     # section 7 (skipped until decoder exists)
    test_api.py                  # section 6 (skipped until API exists)
  perf/
    test_latency_memory.py       # section 9, marker: perf
  test_io.py, test_pipeline.py, test_evaluation.py   # section 4 unit gaps
pytest.ini                       # markers: perf, slow; default excludes perf
.github/workflows/tests.yml      # optional, needs your approval
```

Order of work: (1) `conftest.py` + `pipeline`/`io`/`evaluation` unit tests, (2) model contract tests, (3) e2e + invalid input, (4) reproducibility, (5) perf profiling, (6) Viterbi and API tests once those components land. Default `pytest` stays under ~1 minute; perf and slow tests are opt-in.

## 12. Decisions and information I need

1. **API:** which framework (FastAPI/Flask/other), and the intended request/response shape. Where will the code live?
2. **HMM/Viterbi:** who owns it, and is the transition model based on arc connectivity, route distance, or both?
3. **Which model serves the API?** The Phase 5 report selects Logistic Regression; I will test all five regardless.
4. **Leakage guard:** do you want the owners of Random Forest and MLP to add the same `route`/`label` column check the other three have, or should the integration layer own that check? I will not edit the model files.
5. **Single-class training and zero-row prediction:** which layer is responsible for handling them (matcher vs models)?
6. **CI:** may I add `pytest.ini` and a GitHub Actions workflow, and a constraints/lock file for dependencies?
7. **Profiling hardware:** are the agreed budgets in section 9 acceptable, and who can run the real-data profile locally on the training split?

## 13. Risks

- API and Viterbi tests are written against an assumed contract, so they may need rework when the real code arrives.
- 0% coverage on `pipeline.py` means Phase 3 -> Phase 5 integration has never been verified automatically.
- KNN inference cost and memory may rule it out of any interactive path.
- Unpinned dependencies can break reproducibility between machines.
- Report metrics come from 79 training trips with fold standard deviations of several points, so small differences between models should not drive integration choices on their own.
