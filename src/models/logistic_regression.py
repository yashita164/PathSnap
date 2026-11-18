"""Logistic Regression classifier for candidate-level map-matching data."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_FORBIDDEN_COLUMN_SUBSTRINGS = ("route", "label")


def _check_no_leakage_columns(X: ArrayLike) -> None:
    if isinstance(X, pd.DataFrame):
        forbidden = [
            column
            for column in X.columns
            if any(
                substring in str(column).lower()
                for substring in _FORBIDDEN_COLUMN_SUBSTRINGS
            )
        ]
        if forbidden:
            raise ValueError(
                f"Feature matrix contains forbidden column(s) {forbidden}: "
                "route and label information must not be used as features."
            )


def _binary_labels(y: ArrayLike) -> np.ndarray:
    labels = np.asarray(y).ravel()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y must contain only 0/1 labels.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Training labels must contain both 0 and 1.")
    return labels.astype(np.int64)


def build_model(
    *,
    max_iter: int = 1000,
    C: float = 1.0,
    solver: str = "lbfgs",
    random_state: int | None = 42,
    impute_strategy: str = "median",
) -> Pipeline:
    """Build the imputation, scaling, and Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=impute_strategy,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=max_iter,
                    C=C,
                    solver=solver,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    max_iter: int = 1000,
    C: float = 1.0,
    solver: str = "lbfgs",
    random_state: int | None = 42,
    impute_strategy: str = "median",
) -> Pipeline:
    """Fit the pipeline on training candidates and binary match labels."""
    _check_no_leakage_columns(X_train)
    labels = _binary_labels(y_train)
    if len(X_train) != len(labels):
        raise ValueError(
            f"X and y length mismatch: {len(X_train)} vs {len(labels)}."
        )

    model = build_model(
        max_iter=max_iter,
        C=C,
        solver=solver,
        random_state=random_state,
        impute_strategy=impute_strategy,
    )
    model.fit(X_train, labels)
    return model


def predict_model(model: Pipeline, X: ArrayLike) -> np.ndarray:
    """Return binary class predictions."""
    _check_no_leakage_columns(X)
    return np.asarray(model.predict(X), dtype=np.int64)


def predict_proba_model(model: Pipeline, X: ArrayLike) -> np.ndarray:
    """Return predicted probabilities for the positive class (class 1)."""
    _check_no_leakage_columns(X)
    classifier = model.named_steps["logistic_regression"]
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise ValueError("The Logistic Regression model has not been fitted.")
    positive_class = np.flatnonzero(np.asarray(classes) == 1)
    if positive_class.size == 0:
        raise ValueError("The trained model has no class 1 probability.")

    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    return probabilities[:, int(positive_class[0])]
