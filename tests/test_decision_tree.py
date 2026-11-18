import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURE_COLUMNS
from src.models.decision_tree import (
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
    assert list(model.named_steps) == ["imputer", "decision_tree"]
    classifier = model.named_steps["decision_tree"]
    assert isinstance(classifier, DecisionTreeClassifier)
    assert (
        classifier.criterion,
        classifier.max_depth,
        classifier.min_samples_split,
        classifier.min_samples_leaf,
        classifier.random_state,
    ) == ("gini", 10, 2, 2, 42)
    assert not any(
        isinstance(step, StandardScaler) for step in model.named_steps.values()
    )


def test_hyperparameters_are_configurable() -> None:
    X, y = _candidate_data()

    model = train_model(
        X,
        y,
        criterion="entropy",
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=3,
        random_state=7,
    )
    classifier = model.named_steps["decision_tree"]

    assert (
        classifier.criterion,
        classifier.max_depth,
        classifier.min_samples_split,
        classifier.min_samples_leaf,
        classifier.random_state,
    ) == ("entropy", 5, 4, 3, 7)


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


def test_route_or_label_columns_are_rejected() -> None:
    X, y = _candidate_data()

    with pytest.raises(ValueError, match="route and label"):
        train_model(X.assign(route_arc_id=0), y)
