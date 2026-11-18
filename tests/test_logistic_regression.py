import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS
from src.models.logistic_regression import (
    build_model,
    predict_model,
    predict_proba_model,
    train_model,
)


def _candidate_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(100, len(FEATURE_COLUMNS)))
    labels = (values[:, 0] + values[:, 1] > 0).astype(np.int64)
    return pd.DataFrame(values, columns=FEATURE_COLUMNS), labels


def test_model_builds_with_expected_defaults() -> None:
    model = build_model()

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == [
        "imputer",
        "scaler",
        "logistic_regression",
    ]
    classifier = model.named_steps["logistic_regression"]
    assert isinstance(classifier, LogisticRegression)
    assert (classifier.max_iter, classifier.C, classifier.solver) == (
        1000,
        1.0,
        "lbfgs",
    )
    assert classifier.random_state == 42
    assert isinstance(model.named_steps["scaler"], StandardScaler)


def test_hyperparameters_are_configurable() -> None:
    X, y = _candidate_data()

    model = train_model(X, y, max_iter=500, C=0.5, solver="liblinear", random_state=7)
    classifier = model.named_steps["logistic_regression"]

    assert (classifier.max_iter, classifier.C, classifier.solver) == (
        500,
        0.5,
        "liblinear",
    )
    assert classifier.random_state == 7


def test_training_predictions_and_positive_class_probabilities() -> None:
    X, y = _candidate_data()

    model = train_model(X, y)
    predictions = predict_model(model, X)
    probabilities = predict_proba_model(model, X)

    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (len(X),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_missing_values_are_imputed_and_empty_columns_preserved() -> None:
    X, y = _candidate_data()
    X.loc[0, FEATURE_COLUMNS[0]] = np.nan
    X[FEATURE_COLUMNS[-1]] = np.nan

    model = train_model(X, y)
    probabilities = predict_proba_model(model, X)

    assert probabilities.shape == (len(X),)
    assert np.isfinite(probabilities).all()
    assert model.named_steps["imputer"].statistics_.shape == (
        len(FEATURE_COLUMNS),
    )


def test_scaler_is_fitted_only_on_training_candidates() -> None:
    X, y = _candidate_data()
    X_train = X.iloc[:70]
    X_test = X.iloc[70:].copy() + 1000

    model = train_model(X_train, y[:70])
    scaler_mean = model.named_steps["scaler"].mean_.copy()
    predict_proba_model(model, X_test)

    np.testing.assert_allclose(scaler_mean, X_train.mean().to_numpy())
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, scaler_mean)


def test_route_or_label_columns_are_rejected() -> None:
    X, y = _candidate_data()

    with pytest.raises(ValueError, match="route and label"):
        train_model(X.assign(route_arc_id=0), y)
