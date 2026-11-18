import json

import joblib
import numpy as np
import pandas as pd
import pytest

from api.config import Settings
from api.model_loader import clear_model_cache, load_classifier
from src.features import FEATURE_COLUMNS
from src.optimization.optimized_logistic_regression import (
    load_optimized_parameters,
    predict_optimized_proba,
    train_optimized_model,
)


def _candidate_data(seed: int = 121) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(120, len(FEATURE_COLUMNS)))
    labels = (values[:, 0] + 0.5 * values[:, 1] > 0).astype(np.int64)
    features = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    features.loc[[2, 17], FEATURE_COLUMNS[0]] = np.nan
    return features, labels


def test_selected_configuration_is_loaded_and_applied() -> None:
    X, y = _candidate_data()

    parameters = load_optimized_parameters()
    model = train_optimized_model(X, y)

    classifier = model.named_steps["logistic_regression"]
    assert parameters["C"] == pytest.approx(2.778663410880327)
    assert classifier.C == parameters["C"]
    assert classifier.max_iter == 1000
    assert classifier.solver == "lbfgs"
    assert classifier.random_state == 42
    assert list(model.named_steps) == ["imputer", "scaler", "logistic_regression"]
    assert model.named_steps["imputer"].strategy == "median"


def test_prediction_probabilities_preserve_class_order_and_row_alignment() -> None:
    X, y = _candidate_data()
    model = train_optimized_model(X.iloc[:90], y[:90])
    X_test = X.iloc[90:]

    native_probabilities = model.predict_proba(X_test)
    class1_probabilities = predict_optimized_proba(model, X_test)

    assert model.named_steps["logistic_regression"].classes_.tolist() == [0, 1]
    assert native_probabilities.shape == (len(X_test), 2)
    assert class1_probabilities.shape == (len(X_test),)
    assert np.isfinite(native_probabilities).all()
    assert np.all((native_probabilities >= 0.0) & (native_probabilities <= 1.0))
    np.testing.assert_allclose(native_probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(class1_probabilities, native_probabilities[:, 1])

    order = np.array([4, 1, 8, 2, 0])
    np.testing.assert_allclose(
        predict_optimized_proba(model, X_test.iloc[order]),
        class1_probabilities[order],
    )


def test_preprocessing_is_fitted_once_and_reused_for_prediction() -> None:
    X, y = _candidate_data()
    X_train = X.iloc[:90]
    X_test = X.iloc[90:].copy() + 1000.0
    model = train_optimized_model(X_train, y[:90])
    imputer_statistics = model.named_steps["imputer"].statistics_.copy()
    scaler_mean = model.named_steps["scaler"].mean_.copy()
    scaler_scale = model.named_steps["scaler"].scale_.copy()

    probabilities = predict_optimized_proba(model, X_test)

    assert probabilities.shape == (len(X_test),)
    np.testing.assert_array_equal(
        model.named_steps["imputer"].statistics_, imputer_statistics
    )
    np.testing.assert_array_equal(model.named_steps["scaler"].mean_, scaler_mean)
    np.testing.assert_array_equal(model.named_steps["scaler"].scale_, scaler_scale)


def test_feature_columns_must_match_canonical_order() -> None:
    X, y = _candidate_data()

    with pytest.raises(ValueError, match="canonical order"):
        train_optimized_model(X.loc[:, list(reversed(FEATURE_COLUMNS))], y)

    model = train_optimized_model(X, y)
    with pytest.raises(ValueError, match="canonical order"):
        predict_optimized_proba(
            model, X.loc[:, list(reversed(FEATURE_COLUMNS))]
        )


def test_configuration_errors_are_reported(tmp_path) -> None:
    config_path = tmp_path / "bad_parameters.json"
    config_path.write_text(
        json.dumps({"model": "Logistic Regression", "parameters": {"C": 1.0}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="parameters must be exactly"):
        load_optimized_parameters(config_path)


def test_fitted_pipeline_can_be_persisted_for_api_probability_loading(tmp_path) -> None:
    X, y = _candidate_data()
    model = train_optimized_model(X, y)
    expected = predict_optimized_proba(model, X.iloc[:12])
    model_path = tmp_path / "optimized_logistic_regression.joblib"

    joblib.dump(model, model_path)
    settings = Settings(
        model_path=model_path,
        model_name="optimized-logistic-regression",
    )
    try:
        api_classifier = load_classifier(settings)
        np.testing.assert_allclose(api_classifier.predict_proba(X.iloc[:12]), expected)
    finally:
        clear_model_cache()
