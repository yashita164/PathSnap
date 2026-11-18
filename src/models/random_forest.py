"""Random Forest classifier for candidate-level map-matching data."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_model(
    *,
    n_estimators: int = 200,
    max_depth: int | None = 20,
    min_samples_split: int | float = 2,
    min_samples_leaf: int | float = 2,
    max_features: str | int | float | None = "sqrt",
    random_state: int | None = 42,
    n_jobs: int | None = -1,
) -> Pipeline:
    """Build a missing-value-aware Random Forest classification pipeline."""
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("random_forest", classifier),
        ]
    )


def train_model(
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    n_estimators: int = 200,
    max_depth: int | None = 20,
    min_samples_split: int | float = 2,
    min_samples_leaf: int | float = 2,
    max_features: str | int | float | None = "sqrt",
    random_state: int | None = 42,
    n_jobs: int | None = -1,
) -> Pipeline:
    """Fit a Random Forest pipeline and return it."""
    model = build_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def predict_model(model: Pipeline, X: ArrayLike) -> np.ndarray:
    """Return binary class predictions."""
    predictions = np.asarray(model.predict(X))
    return predictions


def predict_proba_model(model: Pipeline, X: ArrayLike) -> np.ndarray:
    """Return predicted probabilities for the positive class (class 1)."""
    classifier: ClassifierMixin = model.named_steps["random_forest"]
    classes = np.asarray(classifier.classes_)
    if 1 not in classes:
        raise ValueError("The trained model has no class 1 probability")

    class_index = int(np.flatnonzero(classes == 1)[0])
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    return probabilities[:, class_index]
