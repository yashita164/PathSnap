import numpy as np
from sklearn.pipeline import Pipeline

from src.models.random_forest import (
    build_model,
    predict_model,
    predict_proba_model,
    train_model,
)


def test_model_builds_successfully() -> None:
    model = build_model()

    assert isinstance(model, Pipeline)


def test_training_and_prediction() -> None:
    X = np.array(
        [
            [0.0, 1.0],
            [0.1, 1.1],
            [0.2, 1.2],
            [9.0, 10.0],
            [9.1, 10.1],
            [9.2, 10.2],
        ]
    )
    y = np.array([1, 1, 1, 0, 0, 0])

    model = train_model(X, y)
    predictions = predict_model(model, X)

    assert set(np.unique(predictions)).issubset({0, 1})
    assert predictions.shape == (len(X),)


def test_probabilities_are_valid_and_one_dimensional() -> None:
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([1, 1, 1, 0, 0, 0])

    model = train_model(X, y)
    probabilities = predict_proba_model(model, X)

    assert probabilities.ndim == 1
    assert probabilities.shape == (len(X),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_missing_values_are_imputed() -> None:
    X = np.array(
        [
            [0.0, np.nan],
            [0.5, 1.0],
            [1.0, 1.5],
            [8.0, 9.0],
            [9.0, np.nan],
            [10.0, 11.0],
        ]
    )
    y = np.array([1, 1, 1, 0, 0, 0])

    model = train_model(X, y)
    predictions = predict_model(model, X)

    assert predictions.shape == (len(X),)


def test_common_feature_matrix_formats() -> None:
    X = np.array([[0.0, 1.0], [1.0, 2.0], [8.0, 9.0], [9.0, 10.0]])
    y = np.array([1, 1, 0, 0])

    model = train_model(X, y)

    predictions = predict_model(model, X.tolist())
    assert predictions.shape == (len(X),)


def test_small_dataset_trains_successfully() -> None:
    X = np.array([[0.0], [1.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y, n_estimators=10)

    assert len(predict_model(model, X)) == len(X)
