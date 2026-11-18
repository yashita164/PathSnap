"""Decision Tree classifier for candidate-level map-matching data."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


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
    criterion: str = "gini",
    max_depth: int | None = 10,
    min_samples_split: int | float = 2,
    min_samples_leaf: int | float = 2,
    random_state: int | None = 42,
    impute_strategy: str = "median",
) -> Pipeline:
    """Build the imputation and Decision Tree pipeline."""
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=impute_strategy,
                    keep_empty_features=True,
                ),
            ),
            (
                "decision_tree",
                DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    criterion: str = "gini",
    max_depth: int | None = 10,
    min_samples_split: int | float = 2,
    min_samples_leaf: int | float = 2,
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
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
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
    classifier = model.named_steps["decision_tree"]
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise ValueError("The Decision Tree model has not been fitted.")
    positive_class = np.flatnonzero(np.asarray(classes) == 1)
    if positive_class.size == 0:
        raise ValueError("The trained model has no class 1 probability.")

    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    return probabilities[:, int(positive_class[0])]
