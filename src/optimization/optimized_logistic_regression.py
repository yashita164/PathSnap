"""Recreate the Phase 6 Logistic Regression from its saved configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline

from src.features import FEATURE_COLUMNS
from src.models.logistic_regression import predict_proba_model, train_model


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "optimization"
    / "best_parameters.json"
)
_PARAMETER_NAMES = {
    "C",
    "max_iter",
    "solver",
    "random_state",
    "impute_strategy",
}


def load_optimized_parameters(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the selected Phase 6 classifier parameters."""
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    try:
        configuration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read optimized model configuration: {path}") from error

    if not isinstance(configuration, dict) or configuration.get("model") != "Logistic Regression":
        raise ValueError("Optimized configuration must describe Logistic Regression.")
    parameters = configuration.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != _PARAMETER_NAMES:
        raise ValueError(
            f"Optimized configuration parameters must be exactly "
            f"{sorted(_PARAMETER_NAMES)}."
        )

    C = parameters["C"]
    max_iter = parameters["max_iter"]
    solver = parameters["solver"]
    random_state = parameters["random_state"]
    impute_strategy = parameters["impute_strategy"]
    if isinstance(C, bool) or not isinstance(C, (int, float)) or not np.isfinite(C) or C <= 0:
        raise ValueError("Optimized Logistic Regression C must be finite and positive.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Optimized Logistic Regression max_iter must be a positive integer.")
    if not isinstance(solver, str) or not solver:
        raise ValueError("Optimized Logistic Regression solver must be a non-empty string.")
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, int)
    ):
        raise ValueError("Optimized Logistic Regression random_state must be an integer or null.")
    if not isinstance(impute_strategy, str) or not impute_strategy:
        raise ValueError(
            "Optimized Logistic Regression impute_strategy must be a non-empty string."
        )

    return {
        "C": float(C),
        "max_iter": max_iter,
        "solver": solver,
        "random_state": random_state,
        "impute_strategy": impute_strategy,
    }


def _validate_feature_columns(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Optimized Logistic Regression requires a feature DataFrame.")
    if tuple(X.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Feature columns must exactly match FEATURE_COLUMNS in canonical order."
        )


def train_optimized_model(
    X_train: pd.DataFrame,
    y_train: ArrayLike,
    config_path: str | Path | None = None,
) -> Pipeline:
    """Fit the selected model using only the supplied candidate training rows."""
    _validate_feature_columns(X_train)
    return train_model(X_train, y_train, **load_optimized_parameters(config_path))


def predict_optimized_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return row-aligned class-1 probabilities for canonical feature rows."""
    _validate_feature_columns(X)
    probabilities = predict_proba_model(model, X)
    if probabilities.shape != (len(X),):
        raise ValueError("Optimized Logistic Regression returned an invalid probability shape.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("Optimized Logistic Regression returned invalid probabilities.")
    return probabilities
