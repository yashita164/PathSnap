# PathSnap API

Run from the repository root:

    pip install -r requirements.txt -r requirements-api.txt
    set PATHSNAP_MODEL_PATH=path\to\final_model.joblib      (PowerShell: $env:PATHSNAP_MODEL_PATH="...")
    uvicorn api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, whether a model file is configured, available decoders |
| `GET /trips` | Development trip IDs (student_test only) |
| `POST /predict/{trip_id}` | Candidates -> features -> classifier -> decoder -> arcs + geometry + metrics |

`POST /predict` accepts an optional JSON body: `{"decoder": "argmax", "search_radius_m": 50, "max_candidates": 8}`.

Configuration is by environment variables (see `api/config.py`). The classifier is any
fitted estimator/Pipeline saved with `joblib.dump` that has `predict_proba`; the file name is
not assumed. Without a model file, `/predict` returns 503 `model_unavailable`.

## Plugging in the HMM/Viterbi decoder

    from api.decoding import register_decoder, DecodeInput, DecodeResult

    class HmmViterbiDecoder:
        name = "hmm"
        def decode(self, data: DecodeInput) -> DecodeResult:
            # data.candidates: one row per (observation, arc) incl. features/rank/distance
            # data.probabilities: classifier P(match), aligned with data.candidates rows
            # data.track / data.nodes / data.arcs: for building transition costs
            # return arc id per GPS observation (-1 = unmatched) and a score
            ...

    register_decoder("hmm", HmmViterbiDecoder)

Then call with `{"decoder": "hmm"}` or set `PATHSNAP_DECODER=hmm`.

## Data policy built in

Only `data/student_test/` is served. `.route` files and `src.io.load_trip` are never used, and no
response field can carry reference routes or labels. Evaluator data is unreachable.
