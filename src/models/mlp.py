"""MLP classifier for candidate-level map-matching data."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_PARAMS: dict[str, Any] = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "max_iter": 300,
    "early_stopping": True,
    "random_state": 42,
}


def build_model(
    *,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    activation: str = "relu",
    solver: str = "adam",
    alpha: float = 0.0001,
    learning_rate_init: float = 0.001,
    max_iter: int = 300,
    early_stopping: bool = True,
    random_state: int | None = 42,
    **mlp_kwargs: Any,
) -> Pipeline:
    """Build an imputer -> scaler -> MLP pipeline.

    Extra keyword arguments (e.g. ``batch_size``, ``n_iter_no_change``) are
    forwarded to ``sklearn.neural_network.MLPClassifier`` so the model can be
    tuned later without changing this module.

    The pipeline is unfitted; imputation statistics and scaling parameters are
    learned only when ``fit`` is called, i.e. on training data only.
    """
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
        **mlp_kwargs,
    )
    return Pipeline(
        [
            # keep_empty_features keeps column count stable if a training
            # column is entirely missing (it is imputed with 0).
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("mlp", classifier),
        ]
    )


def train_model(
    X_train: ArrayLike,
    y_train: ArrayLike,
    **params: Any,
) -> Pipeline:
    """Fit the MLP pipeline on training data only and return it."""
    model = build_model(**params)
    model.fit(X_train, y_train)
    return model


def predict_model(model: Pipeline, X: ArrayLike) -> np.ndarray:
    """Return binary class predictions."""
    return np.asarray(model.predict(X))


def predict_proba_model(model: Pipeline, X: ArrayLike) -> np.ndarray:
    """Return predicted probabilities for the positive class (class 1)."""
    classes = np.asarray(model.named_steps["mlp"].classes_)
    if 1 not in classes:
        raise ValueError("The trained model has no class 1 probability")

    class_index = int(np.flatnonzero(classes == 1)[0])
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    return probabilities[:, class_index]
