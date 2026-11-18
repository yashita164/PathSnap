"""Classifier loading behind a generic ``predict_proba`` interface.

The API depends only on ``ProbabilityClassifier``: an object whose
``predict_proba(X)`` returns one P(match) per row of the feature DataFrame.
Any fitted scikit-learn estimator/Pipeline (e.g. from src/models/*) that was
saved with ``joblib.dump`` can be used, whatever model won the comparison; the
file is chosen via PATHSNAP_MODEL_PATH. Nothing is trained here.

Security note: joblib files are pickles, so only point PATHSNAP_MODEL_PATH at
files you trust. The path is server-side configuration and is never taken from
a request.
"""

from __future__ import annotations

import threading
from typing import Callable, Protocol

import joblib
import numpy as np
import pandas as pd

from api.config import Settings
from api.errors import ApiError, ClassifierError, ModelUnavailableError


class ProbabilityClassifier(Protocol):
    name: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return a 1-D array of positive-class probabilities, len(X)."""
        ...


class SklearnStyleClassifier:
    """Adapter normalising any ``predict_proba`` implementation.

    Accepts a 1-D array (already positive-class) or the usual (n, n_classes)
    array; in the latter case the column for class 1 is selected using
    ``classes_`` when available. Output is validated to be finite and in [0, 1].
    """

    def __init__(self, estimator: object, name: str) -> None:
        self._estimator = estimator
        self.name = name

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            raw = np.asarray(self._estimator.predict_proba(X), dtype=np.float64)
        except ApiError:
            raise
        except Exception as error:  # estimator-specific failures (shape, dtype, ...)
            raise ClassifierError(
                "The classifier failed to produce probabilities for this trip."
            ) from error

        if raw.ndim == 2:
            classes = getattr(self._estimator, "classes_", None)
            if classes is not None and 1 in list(classes):
                column = list(classes).index(1)
            elif raw.shape[1] == 2:
                column = 1
            else:
                raise ClassifierError("The classifier returned an unexpected probability shape.")
            raw = raw[:, column]
        if raw.shape != (len(X),):
            raise ClassifierError("The classifier returned an unexpected number of probabilities.")
        if not np.isfinite(raw).all() or (raw < -1e-9).any() or (raw > 1 + 1e-9).any():
            raise ClassifierError("The classifier returned probabilities outside [0, 1].")
        return np.clip(raw, 0.0, 1.0)


def _display_name(estimator: object, settings: Settings) -> str:
    if settings.model_name:
        return settings.model_name
    steps = getattr(estimator, "steps", None)
    if steps:
        return f"Pipeline[{type(steps[-1][1]).__name__}]"
    return type(estimator).__name__


_CACHE: dict[tuple[str, int], ProbabilityClassifier] = {}
_LOCK = threading.Lock()


def clear_model_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def model_is_available(settings: Settings) -> bool:
    """Cheap check used by /health: is a model file configured and present?"""
    return settings.model_path is not None and settings.model_path.is_file()


def load_classifier(settings: Settings) -> ProbabilityClassifier:
    """Load (and cache by path + mtime) the configured classifier."""
    path = settings.model_path
    if path is None:
        raise ModelUnavailableError(
            "No classifier is configured. Set PATHSNAP_MODEL_PATH to a joblib file."
        )
    if not path.is_file():
        raise ModelUnavailableError("The configured classifier file was not found.")

    key = (str(path.resolve()), path.stat().st_mtime_ns)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        try:
            estimator = joblib.load(path)
        except Exception as error:
            raise ModelUnavailableError(
                "The configured classifier file could not be loaded."
            ) from error
        if not callable(getattr(estimator, "predict_proba", None)):
            raise ModelUnavailableError(
                "The configured classifier does not provide predict_proba."
            )
        classifier = SklearnStyleClassifier(estimator, _display_name(estimator, settings))
        _CACHE.clear()  # keep at most one model resident
        _CACHE[key] = classifier
        return classifier


def get_classifier_loader() -> Callable[[Settings], ProbabilityClassifier]:
    """FastAPI dependency; override in tests or to swap the loading strategy."""
    return load_classifier
