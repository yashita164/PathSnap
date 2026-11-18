import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.mlp import (
    build_model,
    predict_model,
    predict_proba_model,
    train_model,
)

N_FEATURES = 6


def _synthetic_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic candidate-level matrix: rows = candidates, cols = numeric features."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, N_FEATURES)) + y[:, None] * 1.5
    return X, y


def test_model_builds_successfully() -> None:
    model = build_model()

    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["imputer", "scaler", "mlp"]
    mlp_params = model.named_steps["mlp"].get_params()
    assert mlp_params["hidden_layer_sizes"] == (64, 32)
    assert mlp_params["early_stopping"] is True
    assert mlp_params["random_state"] == 42


def test_parameters_are_configurable() -> None:
    model = build_model(hidden_layer_sizes=(16,), alpha=0.01, batch_size=32)
    params = model.named_steps["mlp"].get_params()

    assert params["hidden_layer_sizes"] == (16,)
    assert params["alpha"] == 0.01
    assert params["batch_size"] == 32


def test_training_and_predictions_are_binary() -> None:
    X, y = _synthetic_data()

    model = train_model(X, y)
    predictions = predict_model(model, X)

    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert (predictions == y).mean() > 0.8


def test_probabilities_valid_and_one_dimensional() -> None:
    X, y = _synthetic_data()
    model = train_model(X, y)

    proba = predict_proba_model(model, X)

    assert proba.shape == (len(X),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_pipeline_handles_missing_values() -> None:
    X, y = _synthetic_data()
    rng = np.random.default_rng(1)
    X_missing = X.copy()
    X_missing[rng.random(X.shape) < 0.1] = np.nan

    model = train_model(X_missing, y)
    predictions = predict_model(model, X_missing)
    proba = predict_proba_model(model, X_missing)

    assert not np.isnan(proba).any()
    assert set(np.unique(predictions)).issubset({0, 1})


def test_all_missing_training_column_keeps_feature_count() -> None:
    X, y = _synthetic_data()
    X[:, 2] = np.nan

    model = train_model(X, y)

    assert predict_proba_model(model, X).shape == (len(X),)


def test_works_with_dataframe_feature_matrix() -> None:
    X, y = _synthetic_data()
    columns = [f"feature_{i}" for i in range(N_FEATURES)]
    X_df = pd.DataFrame(X, columns=columns)

    model = train_model(X_df, pd.Series(y))

    assert predict_model(model, X_df).shape == (len(X_df),)
    assert predict_proba_model(model, X_df.iloc[:10]).shape == (10,)


def test_scaler_fitted_on_training_data_only() -> None:
    X_train, y_train = _synthetic_data(seed=0)
    X_test, _ = _synthetic_data(n=50, seed=5)
    X_test = X_test + 100.0

    model = train_model(X_train, y_train)
    scaler = model.named_steps["scaler"]

    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-8)
    predict_proba_model(model, X_test)
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-8)
