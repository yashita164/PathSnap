import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.models.decision_tree import (
    predict_model as decision_tree_predict,
    predict_proba_model as decision_tree_proba,
    train_model as train_decision_tree,
)
from src.models.knn import (
    predict_model as knn_predict,
    predict_proba_model as knn_proba,
    train_model as train_knn,
)
from src.models.logistic_regression import (
    predict_model as logistic_predict,
    predict_proba_model as logistic_proba,
    train_model as train_logistic,
)
from src.models.mlp import (
    predict_model as mlp_predict,
    predict_proba_model as mlp_proba,
    train_model as train_mlp,
)
from src.models.random_forest import (
    predict_model as random_forest_predict,
    predict_proba_model as random_forest_proba,
    train_model as train_random_forest,
)


_MODEL_SPECS = [
    (
        "Logistic Regression",
        train_logistic,
        logistic_predict,
        logistic_proba,
        {},
        "logistic_regression",
    ),
    (
        "Decision Tree",
        train_decision_tree,
        decision_tree_predict,
        decision_tree_proba,
        {},
        "decision_tree",
    ),
    (
        "Random Forest",
        train_random_forest,
        random_forest_predict,
        random_forest_proba,
        {"n_estimators": 12, "n_jobs": 1},
        "random_forest",
    ),
    (
        "MLP",
        train_mlp,
        mlp_predict,
        mlp_proba,
        {
            "hidden_layer_sizes": (8,),
            "max_iter": 100,
            "early_stopping": False,
            "random_state": 42,
        },
        "mlp",
    ),
    (
        "KNN",
        train_knn,
        knn_predict,
        knn_proba,
        {},
        "knn",
    ),
]


def _synthetic_candidates() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20261118)
    values = rng.normal(size=(80, len(FEATURE_COLUMNS)))
    labels = (values[:, 0] + values[:, 1] > 0.0).astype(np.int64)
    return pd.DataFrame(values, columns=FEATURE_COLUMNS), labels


@pytest.fixture(
    scope="module",
    params=_MODEL_SPECS,
    ids=["logistic-regression", "decision-tree", "random-forest", "mlp", "knn"],
)
def fitted_model(request):
    name, train, predict, predict_proba, build_kwargs, step_name = request.param
    X, y = _synthetic_candidates()
    model = train(X, y, **build_kwargs)
    return name, train, predict, predict_proba, build_kwargs, step_name, model, X, y


def test_model_probabilities_are_aligned_and_valid(fitted_model) -> None:
    (
        _,
        _,
        predict,
        predict_proba,
        _,
        step_name,
        model,
        X,
        _,
    ) = fitted_model
    probabilities = predict_proba(model, X)
    predictions = predict(model, X)
    classes = (
        model.classes_
        if step_name == "knn"
        else model.named_steps[step_name].classes_
    )

    assert np.asarray(classes).tolist() == [0, 1]
    assert probabilities.shape == (len(X),)
    assert probabilities.dtype == np.float64
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)).issubset({0, 1})

    permutation = np.random.default_rng(99).permutation(len(X))
    shuffled_probabilities = predict_proba(model, X.iloc[permutation])
    np.testing.assert_allclose(
        shuffled_probabilities,
        probabilities[permutation],
        atol=1e-6 if fitted_model[0] == "KNN" else 0.0,
    )


def test_models_impute_missing_prediction_values(fitted_model) -> None:
    _, _, _, predict_proba, _, _, model, X, _ = fitted_model
    X_missing = X.iloc[:12].copy()
    X_missing.iloc[0, 0] = np.nan

    probabilities = predict_proba(model, X_missing)

    assert probabilities.shape == (len(X_missing),)
    assert np.isfinite(probabilities).all()


def test_model_specific_infinity_handling(fitted_model) -> None:
    name, _, _, predict_proba, _, _, model, X, _ = fitted_model
    X_infinite = X.iloc[:12].copy()
    X_infinite.iloc[0, 0] = np.inf

    if name == "KNN":
        probabilities = predict_proba(model, X_infinite)
        assert np.isfinite(probabilities).all()
    else:
        with pytest.raises(ValueError):
            predict_proba(model, X_infinite)


def test_zero_row_prediction_requires_caller_to_skip_empty_groups(
    fitted_model,
) -> None:
    _, _, _, predict_proba, _, _, model, X, _ = fitted_model

    with pytest.raises(ValueError):
        predict_proba(model, X.iloc[:0])


@pytest.mark.parametrize(
    ("name", "train", "build_kwargs"),
    [
        pytest.param("Logistic Regression", train_logistic, {}, id="logistic"),
        pytest.param("Decision Tree", train_decision_tree, {}, id="tree"),
        pytest.param(
            "Random Forest",
            train_random_forest,
            {"n_estimators": 4, "n_jobs": 1},
            marks=pytest.mark.xfail(
                strict=True, reason="Random Forest lacks a route/label leakage guard"
            ),
            id="random-forest",
        ),
        pytest.param(
            "MLP",
            train_mlp,
            {
                "hidden_layer_sizes": (4,),
                "max_iter": 30,
                "early_stopping": False,
                "random_state": 42,
            },
            marks=pytest.mark.xfail(
                strict=True, reason="MLP lacks a route/label leakage guard"
            ),
            id="mlp",
        ),
        pytest.param("KNN", train_knn, {}, id="knn"),
    ],
)
def test_training_rejects_route_or_label_features(
    name: str,
    train,
    build_kwargs: dict,
) -> None:
    del name
    X, y = _synthetic_candidates()

    with pytest.raises(ValueError, match="route|label"):
        train(X.assign(route_arc_id=0), y, **build_kwargs)


@pytest.mark.parametrize(
    ("name", "train", "build_kwargs"),
    [
        pytest.param("Logistic Regression", train_logistic, {}, id="logistic"),
        pytest.param("Decision Tree", train_decision_tree, {}, id="tree"),
        pytest.param(
            "Random Forest",
            train_random_forest,
            {"n_estimators": 4, "n_jobs": 1},
            marks=pytest.mark.xfail(
                strict=True, reason="Random Forest lacks binary-label validation"
            ),
            id="random-forest",
        ),
        pytest.param(
            "MLP",
            train_mlp,
            {
                "hidden_layer_sizes": (4,),
                "max_iter": 30,
                "early_stopping": False,
                "random_state": 42,
            },
            marks=pytest.mark.xfail(
                strict=True, reason="MLP lacks binary-label validation"
            ),
            id="mlp",
        ),
        pytest.param("KNN", train_knn, {}, id="knn"),
    ],
)
def test_training_rejects_non_binary_labels(
    name: str,
    train,
    build_kwargs: dict,
) -> None:
    del name
    X, y = _synthetic_candidates()

    with pytest.raises(ValueError, match="0/1|0 and 1|binary"):
        train(X, y * 2, **build_kwargs)


def test_knn_single_row_batching_roundoff_stays_bounded_and_preserves_classes() -> None:
    X, y = _synthetic_candidates()
    model = train_knn(X.to_numpy(), y)
    full_probabilities = knn_proba(model, X.to_numpy())
    one_row_batches = knn_proba(model, X.to_numpy(), batch_size=1)

    np.testing.assert_allclose(full_probabilities, one_row_batches, atol=1e-6)
    np.testing.assert_array_equal(
        knn_predict(model, X.to_numpy()),
        (one_row_batches >= 0.5).astype(np.int64),
    )
