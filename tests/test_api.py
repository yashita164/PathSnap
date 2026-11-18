"""API tests using synthetic trips written in the repository's on-disk format.

No real data, evaluator data or trained project model is used: classifiers are
small stand-ins that compute probabilities from the real feature columns.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.io
from api import decoding
from api.config import Settings, get_settings
from api.decoding import ArgmaxDecoder, DecodeInput, DecodeResult
from api.errors import ConfigurationError
from api.main import create_app
from api.model_loader import (
    SklearnStyleClassifier,
    clear_model_cache,
    get_classifier_loader,
)
from api.schemas import PredictResponse
from src.features import FEATURE_COLUMNS
from src.hmm import calculate_route_continuity, decode_route

# --- synthetic road network (same geometry style as tests/test_phase3.py) ---
NODES = [(-0.001, 0.0), (0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.002, 0.001)]
ARCS = [(0, 1), (1, 2), (2, 3), (3, 4)]
# lon, lat, timestamp. Nearest arcs: 0, 0, 1, 2, 2, 3
TRACK = [
    (-0.0005, 0.00005, 1.0),
    (-0.0003, 0.00004, 2.0),
    (0.0005, 0.00005, 3.0),
    (0.00099, 0.0005, 4.0),
    (0.00101, 0.0007, 5.0),
    (0.0015, 0.00105, 6.0),
]
FAR_POINT = (1.0, 1.0)


def write_trip(split_dir: Path, trip_id: str, track=TRACK, with_route=False) -> Path:
    directory = split_dir / trip_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{trip_id}.track").write_text(
        "\n".join(f"{lon:.8f} {lat:.8f} {ts:.1f}" for lon, lat, ts in track) + "\n"
    )
    (directory / f"{trip_id}.nodes").write_text(
        "\n".join(f"{lon:.8f} {lat:.8f}" for lon, lat in NODES) + "\n"
    )
    (directory / f"{trip_id}.arcs").write_text(
        "\n".join(f"{a} {b}" for a, b in ARCS) + "\n"
    )
    if with_route:
        (directory / f"{trip_id}.route").write_text("0\n1\n2\n3\n")
    return directory


class DistanceStub:
    """Stand-in classifier: probability decays with the real distance feature."""

    classes_ = np.array([0, 1])

    def __init__(self):
        self.calls: list[list[str]] = []

    def predict_proba(self, X):
        self.calls.append(list(X.columns))
        p1 = np.exp(-X["point_to_segment_distance_m"].to_numpy(dtype=float) / 20.0)
        return np.column_stack([1.0 - p1, p1])


@pytest.fixture(autouse=True)
def _clean_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    write_trip(root / "student_test", "trip_b", with_route=True)
    write_trip(root / "student_test", "trip_a")
    # Splits that must never be served.
    write_trip(root / "train", "trip_train", with_route=True)
    write_trip(root / "evaluator_test", "trip_eval")
    return root


def make_client(settings, classifier=None):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    if classifier is not None:
        app.dependency_overrides[get_classifier_loader] = lambda: (
            lambda _settings: SklearnStyleClassifier(classifier, "stub-model")
        )
    return TestClient(app)


@pytest.fixture
def settings(data_root):
    return Settings(data_root=data_root)


@pytest.fixture
def stub():
    return DistanceStub()


@pytest.fixture
def client(settings, stub):
    return make_client(settings, stub)


def all_keys(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield str(key)
            yield from all_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from all_keys(inner)


FORBIDDEN_KEY_PARTS = ("route", "label", "truth", "reference", "is_match", "evaluator")


# ------------------------------------------------------------------- health
def test_health_without_model(settings):
    response = make_client(settings).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_available"] is False
    assert body["default_decoder"] == "argmax"
    assert "argmax" in body["available_decoders"]
    assert "hmm" in body["available_decoders"]
    assert body["dev_split"] == "student_test"


def test_health_reports_model_file(settings, tmp_path):
    model_file = tmp_path / "model.joblib"
    joblib.dump(DistanceStub(), model_file)
    configured = Settings(data_root=settings.data_root, model_path=model_file)
    assert make_client(configured).get("/health").json()["model_available"] is True


# -------------------------------------------------------------------- trips
def test_trips_lists_only_development_split(client):
    response = client.get("/trips")
    assert response.status_code == 200
    body = response.json()
    assert body == {"split": "student_test", "count": 2, "trips": ["trip_a", "trip_b"]}
    assert not any(p in k.lower() for k in all_keys(body) for p in FORBIDDEN_KEY_PARTS)


def test_trips_empty_when_split_directory_missing(tmp_path):
    empty = Settings(data_root=tmp_path / "nowhere")
    body = make_client(empty).get("/trips").json()
    assert body["count"] == 0 and body["trips"] == []


# ------------------------------------------------------------------ predict
def test_predict_success_structure_and_values(client, stub):
    response = client.post("/predict/trip_a")
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "trip_id", "status", "model", "decoder", "predicted_arc_ids",
        "gps_points", "geometry", "matches", "metrics",
    }
    assert body["trip_id"] == "trip_a"
    assert body["status"] == "ok"
    assert body["model"] == "stub-model"
    assert body["decoder"] == "argmax"
    assert len(body["gps_points"]) == len(TRACK)
    assert body["gps_points"][0] == {
        "gps_observation_index": 0,
        "longitude": TRACK[0][0],
        "latitude": TRACK[0][1],
    }
    # Per-observation arcs are [0, 0, 1, 2, 2, 3]; consecutive duplicates collapse.
    assert [m["arc_id"] for m in body["matches"]] == [0, 0, 1, 2, 2, 3]
    assert body["predicted_arc_ids"] == [0, 1, 2, 3]
    for match in body["matches"]:
        assert 0.0 <= match["probability"] <= 1.0
        assert match["candidate_count"] >= 1

    geometry = body["geometry"]
    assert geometry["type"] == "FeatureCollection"
    assert [f["properties"]["arc_id"] for f in geometry["features"]] == [0, 1, 2, 3]
    assert [f["properties"]["sequence_index"] for f in geometry["features"]] == [0, 1, 2, 3]
    first = geometry["features"][0]["geometry"]
    assert first["type"] == "LineString"
    assert first["coordinates"] == [[-0.001, 0.0], [0.0, 0.0]]

    metrics = body["metrics"]
    assert metrics["n_gps_observations"] == 6
    assert metrics["n_matched_observations"] == 6
    assert metrics["match_coverage"] == 1.0
    assert metrics["n_predicted_arcs"] == 4
    assert metrics["n_candidates"] >= 6
    assert 0.0 < metrics["min_match_probability"] <= metrics["mean_match_probability"] <= 1.0
    assert set(metrics["timings_ms"]) == {
        "load", "candidates", "features", "classify", "decode", "total",
    }

    # The classifier received exactly the agreed feature columns, nothing else.
    assert stub.calls and all(call == list(FEATURE_COLUMNS) for call in stub.calls)


def test_predict_accepts_request_options(client):
    body = client.post(
        "/predict/trip_a", json={"max_candidates": 1, "search_radius_m": 80}
    ).json()
    assert body["metrics"]["max_candidates"] == 1
    assert body["metrics"]["search_radius_m"] == 80
    assert body["metrics"]["n_candidates"] == body["metrics"]["n_gps_observations"]


def test_response_never_exposes_reference_or_ground_truth(client):
    for trip_id in ("trip_a", "trip_b"):  # trip_b has a .route file on disk
        body = client.post(f"/predict/{trip_id}").json()
        offending = [
            key for key in all_keys(body)
            if any(part in key.lower() for part in FORBIDDEN_KEY_PARTS)
        ]
        assert offending == []


def test_route_files_are_never_opened(settings, stub, monkeypatch):
    opened: list[str] = []
    original = src.io._read_table

    def spy(path, columns):
        opened.append(str(path))
        return original(path, columns)

    monkeypatch.setattr(src.io, "_read_table", spy)
    monkeypatch.setattr(src.io, "load_route", lambda *a, **k: pytest.fail("route read"))
    monkeypatch.setattr(src.io, "load_trip", lambda *a, **k: pytest.fail("trip read"))

    response = make_client(settings, stub).post("/predict/trip_b")  # has a .route file
    assert response.status_code == 200
    assert opened and not any(path.endswith(".route") for path in opened)


def test_predict_is_computed_not_hardcoded(settings):
    class PreferFarthest(DistanceStub):
        def predict_proba(self, X):
            p1 = 1.0 - np.exp(-X["point_to_segment_distance_m"].to_numpy(dtype=float) / 20.0)
            return np.column_stack([1.0 - p1, p1])

    # Moving the track point next to a node gives several candidates; two
    # classifiers with opposite preferences must be able to disagree.
    root = settings.data_root / "student_test"
    write_trip(root, "trip_node", track=[(0.00001, 0.00004, 1.0), (0.00002, 0.00006, 2.0)])
    near = make_client(settings, DistanceStub()).post("/predict/trip_node").json()
    far = make_client(settings, PreferFarthest()).post("/predict/trip_node").json()
    assert near["status"] == "ok" and far["status"] == "ok"
    assert near["predicted_arc_ids"] != far["predicted_arc_ids"]


def test_two_column_one_column_probabilities_both_supported(settings):
    class OneDim(DistanceStub):
        def predict_proba(self, X):
            return np.exp(-X["point_to_segment_distance_m"].to_numpy(dtype=float) / 20.0)

    body = make_client(settings, OneDim()).post("/predict/trip_a").json()
    assert body["predicted_arc_ids"] == [0, 1, 2, 3]


# ----------------------------------------------------------- invalid trips
def test_unknown_trip_returns_404(client):
    response = client.post("/predict/does_not_exist")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "trip_not_found"


@pytest.mark.parametrize("bad_id", ["bad$id", "bad id", "-leading", "a" * 200])
def test_malformed_trip_id_returns_422(client, bad_id):
    response = client.post(f"/predict/{bad_id}")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_trip_id"


@pytest.mark.parametrize("trip_id", ["trip_train", "trip_eval"])
def test_other_splits_are_not_reachable(client, trip_id):
    response = client.post(f"/predict/{trip_id}")
    assert response.status_code == 404


def test_evaluator_split_cannot_be_configured(data_root, monkeypatch):
    with pytest.raises(ConfigurationError):
        Settings(data_root=data_root, dev_split="evaluator_test")
    monkeypatch.setenv("PATHSNAP_DEV_SPLIT", "evaluator_test")
    with pytest.raises(ConfigurationError):
        get_settings()


# ------------------------------------------------------ missing / bad files
@pytest.mark.parametrize("extension", ["nodes", "arcs", "track"])
def test_missing_trip_file_gives_meaningful_error(settings, stub, tmp_path, extension):
    directory = settings.data_root / "student_test" / "trip_a"
    if extension == "track":
        # discovery needs the .track file, so a missing track means unknown trip
        (directory / "trip_a.track").unlink()
        response = make_client(settings, stub).post("/predict/trip_a")
        assert response.status_code == 404
        return
    (directory / f"trip_a.{extension}").unlink()
    response = make_client(settings, stub).post("/predict/trip_a")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "trip_data_incomplete"
    assert f".{extension}" in detail["message"]
    assert str(tmp_path) not in response.text  # no server paths leaked


def test_corrupt_trip_file_gives_422_without_path_leak(settings, stub, tmp_path):
    arcs = settings.data_root / "student_test" / "trip_a" / "trip_a.arcs"
    arcs.write_text("0 not_a_number\n")
    response = make_client(settings, stub).post("/predict/trip_a")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "trip_data_invalid"
    assert str(tmp_path) not in response.text


def test_empty_trip_file_gives_422(settings, stub):
    nodes = settings.data_root / "student_test" / "trip_a" / "trip_a.nodes"
    nodes.write_text("")
    response = make_client(settings, stub).post("/predict/trip_a")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "trip_data_invalid"


# ------------------------------------------------------------ no candidates
def test_no_candidates_is_handled(settings):
    write_trip(
        settings.data_root / "student_test", "trip_far",
        track=[(*FAR_POINT, 1.0), (1.0001, 1.0001, 2.0)],
    )
    stub = DistanceStub()
    response = make_client(settings, stub).post("/predict/trip_far")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_candidates"
    assert body["predicted_arc_ids"] == []
    assert body["geometry"] == {"type": "FeatureCollection", "features": []}
    assert all(m["arc_id"] is None and m["probability"] is None for m in body["matches"])
    assert body["metrics"]["n_candidates"] == 0
    assert body["metrics"]["match_coverage"] == 0.0
    assert body["metrics"]["mean_match_probability"] is None
    assert stub.calls == []  # classifier not invoked with an empty matrix


def test_partial_coverage_when_some_points_have_no_candidates(settings, stub):
    write_trip(
        settings.data_root / "student_test", "trip_mixed",
        track=[(-0.0005, 0.00005, 1.0), (*FAR_POINT, 2.0), (0.0005, 0.00005, 3.0)],
    )
    body = make_client(settings, stub).post("/predict/trip_mixed").json()
    assert body["status"] == "partial"
    assert [m["arc_id"] for m in body["matches"]] == [0, None, 1]
    assert body["predicted_arc_ids"] == [0, 1]
    assert body["metrics"]["n_matched_observations"] == 2
    assert body["metrics"]["match_coverage"] == pytest.approx(2 / 3)


# -------------------------------------------------------- model availability
def test_model_not_configured_returns_503(settings):
    response = make_client(settings).post("/predict/trip_a")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_unavailable"


def test_model_file_missing_returns_503(settings, tmp_path):
    configured = Settings(data_root=settings.data_root, model_path=tmp_path / "gone.joblib")
    response = make_client(configured).post("/predict/trip_a")
    assert response.status_code == 503
    assert str(tmp_path) not in response.text


def test_unloadable_model_file_returns_503(settings, tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_text("this is not a joblib file")
    configured = Settings(data_root=settings.data_root, model_path=bad)
    assert make_client(configured).post("/predict/trip_a").status_code == 503


def test_object_without_predict_proba_returns_503(settings, tmp_path):
    path = tmp_path / "dict.joblib"
    joblib.dump({"not": "a model"}, path)
    configured = Settings(data_root=settings.data_root, model_path=path)
    response = make_client(configured).post("/predict/trip_a")
    assert response.status_code == 503
    assert "predict_proba" in response.json()["detail"]["message"]


def test_model_loaded_from_configured_path_end_to_end(settings, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(DistanceStub(), path)
    configured = Settings(
        data_root=settings.data_root, model_path=path, model_name="my-final-model"
    )
    body = make_client(configured).post("/predict/trip_a").json()
    assert body["model"] == "my-final-model"
    assert body["predicted_arc_ids"] == [0, 1, 2, 3]


def test_unknown_trip_reported_before_model_problem(settings):
    response = make_client(settings).post("/predict/does_not_exist")  # no model either
    assert response.status_code == 404


# ------------------------------------------------------ classifier failures
@pytest.mark.parametrize(
    "bad_output",
    [
        lambda X: np.zeros(len(X) + 1),
        lambda X: np.full(len(X), np.nan),
        lambda X: np.full(len(X), 1.5),
        lambda X: np.zeros((len(X), 3)),
    ],
    ids=["wrong_length", "nan", "out_of_range", "three_columns"],
)
def test_invalid_classifier_output_returns_clean_500(settings, bad_output):
    class Bad:
        def predict_proba(self, X):
            return bad_output(X)

    response = make_client(settings, Bad()).post("/predict/trip_a")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "classifier_error"


def test_classifier_exception_returns_clean_500(settings):
    class Boom:
        def predict_proba(self, X):
            raise RuntimeError("secret internals /srv/model.pkl")

    response = make_client(settings, Boom()).post("/predict/trip_a")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "classifier_error"
    assert "secret" not in response.text


# ------------------------------------------------------- decoder replacement
def test_custom_decoder_can_be_plugged_in(settings, stub, monkeypatch):
    seen = {}

    class AlwaysArcZero:
        name = "always_zero"

        def decode(self, data: DecodeInput) -> DecodeResult:
            seen["n_observations"] = data.n_observations
            seen["n_candidates"] = len(data.candidates)
            seen["probabilities_aligned"] = len(data.probabilities) == len(data.candidates)
            seen["has_network"] = len(data.nodes) == len(NODES) and len(data.arcs) == len(ARCS)
            return DecodeResult(
                np.zeros(data.n_observations, dtype=np.int64),
                np.full(data.n_observations, 0.5),
            )

    monkeypatch.setitem(decoding._DECODERS, "always_zero", AlwaysArcZero)
    body = make_client(settings, stub).post(
        "/predict/trip_a", json={"decoder": "always_zero"}
    ).json()
    assert body["decoder"] == "always_zero"
    assert body["predicted_arc_ids"] == [0]
    assert seen == {
        "n_observations": 6,
        "n_candidates": seen["n_candidates"],
        "probabilities_aligned": True,
        "has_network": True,
    }


def test_hmm_decoder_integrates_with_api_and_response_schema(settings, stub, monkeypatch):
    class HMMDecoder:
        name = "hmm_test"

        def decode(self, data: DecodeInput) -> DecodeResult:
            result = decode_route(
                data.track,
                data.nodes,
                data.arcs,
                data.candidates,
                data.probabilities,
            )
            return DecodeResult(
                arc_ids=result.predictions["hmm_arc_id"].to_numpy(dtype=np.int64),
                scores=result.predictions["hmm_candidate_probability"].to_numpy(
                    dtype=np.float64
                ),
            )

    monkeypatch.setitem(decoding._DECODERS, "hmm_test", HMMDecoder)
    response = make_client(settings, stub).post(
        "/predict/trip_a", json={"decoder": "hmm_test"}
    )

    assert response.status_code == 200
    result = PredictResponse.model_validate(response.json())
    assert result.decoder == "hmm_test"
    assert len(result.matches) == len(TRACK)
    assert result.metrics.n_matched_observations == len(TRACK)
    assert calculate_route_continuity(
        result.predicted_arc_ids,
        pd.DataFrame(ARCS, columns=["from_node", "to_node"]),
    ) == 1.0
    assert all(
        match.probability is not None and 0.0 <= match.probability <= 1.0
        for match in result.matches
    )


def test_default_decoder_is_configurable(settings, stub, monkeypatch):
    class AlwaysArcOne:
        name = "one"

        def decode(self, data):
            return DecodeResult(
                np.ones(data.n_observations, dtype=np.int64),
                np.full(data.n_observations, 0.9),
            )

    monkeypatch.setitem(decoding._DECODERS, "one", AlwaysArcOne)
    configured = Settings(data_root=settings.data_root, decoder="one")
    body = make_client(configured, stub).post("/predict/trip_a").json()
    assert body["decoder"] == "one" and body["predicted_arc_ids"] == [1]


def test_unknown_decoder_returns_422(client):
    response = client.post("/predict/trip_a", json={"decoder": "nope"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unknown_decoder"


@pytest.mark.parametrize(
    "result",
    [
        DecodeResult(np.zeros(3, dtype=np.int64), np.zeros(3)),  # wrong length
        DecodeResult(np.full(6, 99, dtype=np.int64), np.zeros(6)),  # arc out of range
        DecodeResult(np.zeros(6), np.zeros(6)),  # non-integer ids
    ],
    ids=["wrong_length", "out_of_range", "float_ids"],
)
def test_invalid_decoder_output_returns_clean_500(settings, stub, monkeypatch, result):
    class Bad:
        name = "bad"

        def decode(self, data):
            return result

    monkeypatch.setitem(decoding._DECODERS, "bad", Bad)
    response = make_client(settings, stub).post("/predict/trip_a", json={"decoder": "bad"})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "decoder_error"


def test_argmax_decoder_tie_break_is_deterministic():
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0, 0, 0, 2],
            "candidate_arc_id": [7, 3, 5, 9],
            "candidate_rank": [2, 1, 3, 1],
        }
    )
    result = ArgmaxDecoder().decode(
        DecodeInput(
            candidates=candidates,
            probabilities=np.array([0.8, 0.8, 0.1, 0.4]),
            n_observations=4,
            track=pd.DataFrame(),
            nodes=pd.DataFrame(),
            arcs=pd.DataFrame(),
        )
    )
    # obs 0: tie on 0.8 -> lower rank (arc 3); obs 1 and 3 have no candidates.
    assert result.arc_ids.tolist() == [3, -1, 9, -1]
    assert np.isnan(result.scores[[1, 3]]).all()
    assert result.scores[0] == 0.8 and result.scores[2] == 0.4


# ------------------------------------------------------- request validation
@pytest.mark.parametrize(
    "payload",
    [
        {"search_radius_m": -5},
        {"search_radius_m": 10_000},
        {"max_candidates": 0},
        {"decoder": "bad name!"},
        {"route": [0, 1, 2]},  # unknown fields rejected: clients cannot inject routes
    ],
)
def test_invalid_request_body_returns_422(client, payload):
    assert client.post("/predict/trip_a", json=payload).status_code == 422


def test_get_on_predict_is_not_allowed(client):
    assert client.get("/predict/trip_a").status_code == 405
