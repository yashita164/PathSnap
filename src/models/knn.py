"""Distance-weighted KNN classifier for PathSnap candidate-level classification.

Task: given a feature row describing one (GPS observation, candidate road arc)
pair, predict whether the candidate is the correct arc (1) or not (0).

Pipeline: inf->NaN -> SimpleImputer(median) -> StandardScaler -> KNeighborsClassifier

Design notes
------------
* All preprocessing lives inside one sklearn Pipeline, so imputer medians and
  scaler mean/std are learned ONLY from the data passed to ``train_model``.
* No resampling / class re-weighting is done; the class distribution is untouched.
* Labels/route information must never be a feature. ``train_model`` and the
  predict helpers refuse DataFrames that contain a column whose name contains
  "route" or "label" (cheap leakage guard).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Initial hyperparameters (spec).
DEFAULT_N_NEIGHBORS = 5
DEFAULT_WEIGHTS = "distance"
DEFAULT_METRIC = "minkowski"
DEFAULT_P = 2  # Euclidean

_FORBIDDEN_COLUMN_SUBSTRINGS = ("route", "label")


def _inf_to_nan(X):
    """Turn +/-inf into NaN so the imputer handles them (stateless)."""
    X = np.asarray(X, dtype=float)
    return np.where(np.isfinite(X), X, np.nan)


def _check_no_leakage_columns(X) -> None:
    if isinstance(X, pd.DataFrame):
        bad = [c for c in X.columns
               if any(s in str(c).lower() for s in _FORBIDDEN_COLUMN_SUBSTRINGS)]
        if bad:
            raise ValueError(
                f"Feature matrix contains forbidden column(s) {bad}: "
                ".route / label information must not be used as a feature."
            )


def _as_2d_float(X) -> np.ndarray:
    _check_no_leakage_columns(X)
    arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {arr.shape}.")
    return arr


def _as_binary_labels(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    if y.dtype == object:
        y = y.astype(float)
    if np.issubdtype(y.dtype, np.floating) and np.isnan(y).any():
        raise ValueError("y contains NaN.")
    y = y.astype(int)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0/1 labels.")
    return y


def build_model(
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    weights: str = DEFAULT_WEIGHTS,
    metric: str = DEFAULT_METRIC,
    p: int = DEFAULT_P,
    algorithm: str = "auto",
    n_jobs: Optional[int] = None,
    impute_strategy: str = "median",
) -> Pipeline:
    """Construct the (unfitted) preprocessing + KNN pipeline.

    ``p`` is only used when ``metric == "minkowski"`` (p=2 -> Euclidean).
    ``n_jobs=-1`` parallelises neighbour search at inference time.
    ``keep_empty_features=True`` keeps columns that are entirely NaN in the
    training data (imputed to 0 before scaling) so column count never changes.
    """
    return Pipeline(
        steps=[
            ("inf_to_nan", FunctionTransformer(_inf_to_nan, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy=impute_strategy, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, metric=metric, p=p,
                algorithm=algorithm, n_jobs=n_jobs,
            )),
        ]
    )


def train_model(X_train, y_train, **build_kwargs) -> Pipeline:
    """Fit and return the pipeline. ``build_kwargs`` go to ``build_model``.

    * Raises ValueError if y_train does not contain both classes (probability
      output would otherwise be meaningless).
    * If n_samples < n_neighbors, n_neighbors is reduced to n_samples (with a
      warning) instead of failing later at predict time.
    """
    X = _as_2d_float(X_train)
    y = _as_binary_labels(y_train)
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")
    if len(X) == 0:
        raise ValueError("Empty training set.")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"Training labels contain only class {classes.tolist()}; both 0 and 1 "
            "are required to produce positive-class probabilities."
        )

    model = build_model(**build_kwargs)
    k = model.named_steps["knn"].n_neighbors
    if len(X) < k:
        warnings.warn(
            f"Training set has {len(X)} samples < n_neighbors={k}; "
            f"using n_neighbors={len(X)}.",
            UserWarning, stacklevel=2,
        )
        model.set_params(knn__n_neighbors=len(X))

    model.fit(X, y)
    return model


def predict_model(model: Pipeline, X, batch_size: Optional[int] = None) -> np.ndarray:
    """Hard 0/1 class predictions, shape (n_samples,)."""
    proba = predict_proba_model(model, X, batch_size=batch_size)
    return (proba >= 0.5).astype(int)


def predict_proba_model(model: Pipeline, X, batch_size: Optional[int] = None) -> np.ndarray:
    """Positive-class (label 1) probabilities, shape (n_samples,).

    ``batch_size`` predicts in chunks to bound memory on large candidate sets.
    """
    classes = getattr(model, "classes_", None)
    if classes is None or list(classes) != [0, 1]:
        raise ValueError(f"Model must be fitted on classes [0, 1]; got {classes}.")
    arr = _as_2d_float(X)
    if batch_size is None or batch_size >= len(arr):
        return model.predict_proba(arr)[:, 1]
    out = np.empty(len(arr), dtype=float)
    for start in range(0, len(arr), batch_size):
        sl = slice(start, start + batch_size)
        out[sl] = model.predict_proba(arr[sl])[:, 1]
    return out
