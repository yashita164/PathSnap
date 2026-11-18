"""Tests for src/models/knn.py using synthetic candidate-level data only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.knn import (  # noqa: E402
    build_model, train_model, predict_model, predict_proba_model,
)

FEATURES = [
    "d", "rank", "d_minus_best", "d_ratio_best", "frac", "arc_len", "n_within_20m",
    "head_cos", "head_abs_diff", "moving", "speed_prev", "speed_next", "dt_prev",
    "dt_next", "turn_angle", "out_deg_end", "in_deg_start", "has_reverse",
]


def make_data(n=300, seed=0, pos_rate=0.25):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    X = rng.normal(size=(n, len(FEATURES)))
    X[:, 0] += np.where(y == 1, -1.5, 1.5)  # make "d" informative
    return pd.DataFrame(X, columns=FEATURES), y


def test_build_model_defaults():
    m = build_model()
    assert list(m.named_steps) == ["inf_to_nan", "imputer", "scaler", "knn"]
    knn = m.named_steps["knn"]
    assert (knn.n_neighbors, knn.weights, knn.metric, knn.p) == (5, "distance", "minkowski", 2)


def test_configurable_params():
    m = build_model(n_neighbors=9, weights="uniform", metric="manhattan")
    knn = m.named_steps["knn"]
    assert (knn.n_neighbors, knn.weights, knn.metric) == (9, "uniform", "manhattan")


def test_train_predict_outputs():
    X, y = make_data()
    model = train_model(X.iloc[:200], y[:200])
    pred = predict_model(model, X.iloc[200:])
    proba = predict_proba_model(model, X.iloc[200:])
    assert set(np.unique(pred)) <= {0, 1}
    assert pred.shape == (100,)
    assert proba.shape == (100,)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert (predict_model(model, X.iloc[200:]) == (proba >= 0.5)).all()
    assert (pred == y[200:]).mean() > 0.8  # separable synthetic data


def test_numpy_input_and_batching_equal():
    X, y = make_data()
    model = train_model(X.to_numpy(), y)
    full = predict_proba_model(model, X.to_numpy())
    batched = predict_proba_model(model, X.to_numpy(), batch_size=37)
    np.testing.assert_allclose(full, batched)


def test_missing_and_inf_values_handled():
    X, y = make_data()
    X = X.copy()
    rng = np.random.default_rng(1)
    X.iloc[rng.random(X.shape) < 0.1] = np.nan
    X.iloc[0, 1] = np.inf
    X.iloc[1, 2] = -np.inf
    X["dt_next"] = np.nan  # fully-missing column must not crash
    model = train_model(X, y)
    proba = predict_proba_model(model, X)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_preprocessing_fit_on_train_only():
    X, y = make_data()
    X_train, X_test = X.iloc[:200], X.iloc[200:].copy() + 1000  # extreme test shift
    model = train_model(X_train, y[:200])
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, X_train.mean().to_numpy())
    np.testing.assert_allclose(model.named_steps["imputer"].statistics_, X_train.median().to_numpy())
    predict_proba_model(model, X_test)  # must not refit or fail
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, X_train.mean().to_numpy())


def test_small_training_set_reduces_k():
    X, y = make_data(n=300)
    idx = np.array([0, 1, 2])
    while len(np.unique(y[idx])) < 2:  # ensure both classes in the tiny set
        idx[2] += 1
    with pytest.warns(UserWarning):
        model = train_model(X.iloc[idx], y[idx])  # 3 samples < k=5
    assert model.named_steps["knn"].n_neighbors == 3
    proba = predict_proba_model(model, X.iloc[:20])
    assert proba.shape == (20,)


def test_single_class_training_rejected():
    X, _ = make_data()
    with pytest.raises(ValueError, match="both 0 and 1"):
        train_model(X, np.zeros(len(X), dtype=int))


def test_bad_labels_rejected():
    X, y = make_data()
    with pytest.raises(ValueError):
        train_model(X, y + 1)


def test_route_or_label_column_rejected():
    X, y = make_data()
    with pytest.raises(ValueError, match="route"):
        train_model(X.assign(route_arc=1), y)
    with pytest.raises(ValueError):
        train_model(X.assign(label=y), y)


def test_class_distribution_not_changed():
    X, y = make_data(pos_rate=0.1)
    model = train_model(X, y)
    assert model.named_steps["knn"]._fit_X.shape[0] == len(X)
    assert np.bincount(model.named_steps["knn"]._y).tolist() == np.bincount(y).tolist()


def test_unfitted_or_wrong_classes_rejected():
    X, _ = make_data()
    with pytest.raises(ValueError):
        predict_proba_model(build_model(), X)
