# PathSnap API

Run from the repository root in PowerShell:

    .\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

The app serves the frontend at http://127.0.0.1:8000/ and interactive API docs at
http://127.0.0.1:8000/docs. The model path defaults to the project-relative
`artifacts\model.joblib`; no absolute machine path is required.

To supply an external permitted trip dataset, set `PATHSNAP_DATA_ROOT` to a data
root that contains `student_test\<trip_id>\<trip_id>.track`, `.nodes`, and `.arcs`:

    $env:PATHSNAP_DATA_ROOT = "D:\pathsnap-data"

The current process only serves the `student_test` split. It never opens `.route`
files. `PATHSNAP_MODEL_PATH` can override the default model path, and a relative
override is resolved from the repository root:

    $env:PATHSNAP_MODEL_PATH = "artifacts\model.joblib"

The UI selects HMM/Viterbi by default. To make HMM the default for other clients,
set `$env:PATHSNAP_DECODER = "hmm"`; `argmax` remains available.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, whether a model file is configured, available decoders |
| `GET /trips` | Development trip IDs (student_test only) |
| `POST /predict/{trip_id}` | Candidates -> features -> classifier -> decoder -> arcs + geometry + metrics |

`POST /predict` accepts an optional JSON body: `{"decoder": "argmax", "search_radius_m": 50, "max_candidates": 8}`.

Configuration is by environment variables (see `api/config.py`). The classifier is any
fitted estimator/Pipeline saved with `joblib.dump` that has `predict_proba`; the file name is
not assumed. Without a model file, `/predict` returns 503 `model_unavailable`.

The HMM adapter is registered by the application and calls `src.hmm.decode_route`
with candidate probabilities aligned to candidate rows. Each API response
includes GPS observations and the predicted arc geometry; it does not include
reference routes or labels.

## Data policy built in

Only `data/student_test/` is served. `.route` files and `src.io.load_trip` are never used, and no
response field can carry reference routes or labels. Evaluator data is unreachable.
