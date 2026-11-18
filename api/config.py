"""Environment-driven configuration.

Nothing about the final model or decoder is hard-coded: the model file, its
display name and the decoder are all configurable.

Environment variables (all optional):
    PATHSNAP_DATA_ROOT        dataset root                       (default: data)
    PATHSNAP_DEV_SPLIT        development split to serve         (default: student_test)
    PATHSNAP_MODEL_PATH       joblib file of a fitted classifier (default: none -> 503)
    PATHSNAP_MODEL_NAME       display name returned in responses (default: class name)
    PATHSNAP_DECODER          default decoder name               (default: argmax)
    PATHSNAP_SEARCH_RADIUS_M  candidate search radius, metres    (default: 50)
    PATHSNAP_MAX_CANDIDATES   max candidates per GPS point       (default: 8)
    PATHSNAP_CORS_ORIGINS     comma-separated allowed origins    (default: none)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from api.errors import ConfigurationError

# Data policy: only the student development split may be served. Evaluator /
# held-out data must never be reachable through the API, whatever the config says.
ALLOWED_DEV_SPLITS = ("student_test",)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    data_root: Path = Path("data")
    dev_split: str = "student_test"
    model_path: Path | None = None
    model_name: str | None = None
    decoder: str = "argmax"
    search_radius_m: float = 50.0
    max_candidates: int = 8
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dev_split not in ALLOWED_DEV_SPLITS:
            raise ConfigurationError(
                f"dev_split must be one of {list(ALLOWED_DEV_SPLITS)}; "
                "evaluator and other splits are not served by this API."
            )
        if not self.search_radius_m > 0:
            raise ConfigurationError("search_radius_m must be positive.")
        if self.max_candidates < 1:
            raise ConfigurationError("max_candidates must be at least 1.")

    @property
    def split_directory(self) -> Path:
        return self.data_root / self.dev_split


def get_settings() -> Settings:
    """Read settings from the environment (cheap; evaluated per request)."""
    env = os.environ
    try:
        search_radius = float(env.get("PATHSNAP_SEARCH_RADIUS_M", "50"))
        max_candidates = int(env.get("PATHSNAP_MAX_CANDIDATES", "8"))
    except ValueError as error:
        raise ConfigurationError(
            "PATHSNAP_SEARCH_RADIUS_M / PATHSNAP_MAX_CANDIDATES must be numeric."
        ) from error
    model_path = env.get("PATHSNAP_MODEL_PATH")
    data_root = env.get("PATHSNAP_DATA_ROOT")
    origins = tuple(
        origin.strip()
        for origin in env.get("PATHSNAP_CORS_ORIGINS", "").split(",")
        if origin.strip()
    )
    return Settings(
        data_root=(
            _resolve_project_path(data_root)
            if data_root
            else DEFAULT_DATA_ROOT
        ),
        dev_split=env.get("PATHSNAP_DEV_SPLIT", "student_test"),
        model_path=(
            _resolve_project_path(model_path)
            if model_path
            else DEFAULT_MODEL_PATH
        ),
        model_name=env.get("PATHSNAP_MODEL_NAME") or None,
        decoder=env.get("PATHSNAP_DECODER", "argmax"),
        search_radius_m=search_radius,
        max_candidates=max_candidates,
        cors_origins=origins,
    )
