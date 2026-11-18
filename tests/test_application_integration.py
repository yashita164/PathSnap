from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.config import PROJECT_ROOT, Settings, get_settings
from api.main import create_app
from api.model_loader import SklearnStyleClassifier, get_classifier_loader


NODES = [(-0.001, 0.0), (0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.002, 0.001)]
ARCS = [(0, 1), (1, 2), (2, 3), (3, 4)]
TRACK = [
    (-0.0005, 0.00005, 1.0),
    (-0.0003, 0.00004, 2.0),
    (0.0005, 0.00005, 3.0),
    (0.00099, 0.0005, 4.0),
    (0.00101, 0.0007, 5.0),
    (0.0015, 0.00105, 6.0),
]


class ProbabilityStub:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        probability = np.exp(
            -features["point_to_segment_distance_m"].to_numpy(dtype=float) / 20.0
        )
        return np.column_stack((1.0 - probability, probability))


def _write_trip(root: Path, trip_id: str) -> Path:
    directory = root / "student_test" / trip_id
    directory.mkdir(parents=True)
    (directory / f"{trip_id}.track").write_text(
        "\n".join(f"{lon} {lat} {timestamp}" for lon, lat, timestamp in TRACK),
        encoding="utf-8",
    )
    (directory / f"{trip_id}.nodes").write_text(
        "\n".join(f"{lon} {lat}" for lon, lat in NODES),
        encoding="utf-8",
    )
    (directory / f"{trip_id}.arcs").write_text(
        "\n".join(f"{start} {end}" for start, end in ARCS),
        encoding="utf-8",
    )
    return directory


def _client(data_root: Path) -> TestClient:
    app = create_app()
    settings = Settings(data_root=data_root)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_classifier_loader] = lambda: (
        lambda _: SklearnStyleClassifier(ProbabilityStub(), "synthetic-test")
    )
    return TestClient(app)


def test_project_relative_environment_paths_are_resolved(monkeypatch) -> None:
    monkeypatch.setenv("PATHSNAP_MODEL_PATH", r"artifacts\model.joblib")
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", r"external-data")
    monkeypatch.delenv("PATHSNAP_DEV_SPLIT", raising=False)

    settings = get_settings()

    assert settings.model_path == (PROJECT_ROOT / "artifacts" / "model.joblib").resolve()
    assert settings.data_root == (PROJECT_ROOT / "external-data").resolve()
    assert settings.dev_split == "student_test"


def test_default_model_and_data_paths_are_project_relative(monkeypatch) -> None:
    monkeypatch.delenv("PATHSNAP_MODEL_PATH", raising=False)
    monkeypatch.delenv("PATHSNAP_DATA_ROOT", raising=False)
    monkeypatch.delenv("PATHSNAP_DEV_SPLIT", raising=False)

    settings = get_settings()

    assert settings.model_path == (PROJECT_ROOT / "artifacts" / "model.joblib").resolve()
    assert settings.data_root == (PROJECT_ROOT / "data").resolve()
    assert settings.dev_split == "student_test"


def test_frontend_trip_list_and_hmm_prediction_work_end_to_end(tmp_path) -> None:
    data_root = tmp_path / "external-data"
    _write_trip(data_root, "trip_a")
    client = _client(data_root)

    page = client.get("/")
    assert page.status_code == 200
    assert "Predict Route" in page.text
    assert 'fetch("/trips")' in page.text

    health = client.get("/health").json()
    assert health["model_available"] is False
    assert "hmm" in health["available_decoders"]
    trips = client.get("/trips").json()
    assert trips["trips"] == ["trip_a"]

    response = client.post("/predict/trip_a", json={"decoder": "hmm"})

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["decoder"] == "hmm"
    assert len(result["gps_points"]) == len(TRACK)
    assert result["metrics"]["n_matched_observations"] > 0
    assert result["geometry"]["features"]
    assert all(
        len(feature["geometry"]["coordinates"]) == 2
        for feature in result["geometry"]["features"]
    )
