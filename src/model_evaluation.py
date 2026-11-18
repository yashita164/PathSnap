"""Grouped cross-validation and comparison for PathSnap candidate classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.features import FEATURE_COLUMNS
from src.models.decision_tree import (
    predict_model as predict_decision_tree,
    predict_proba_model as predict_decision_tree_proba,
    train_model as train_decision_tree,
)
from src.models.knn import (
    predict_model as predict_knn,
    predict_proba_model as predict_knn_proba,
    train_model as train_knn,
)
from src.models.logistic_regression import (
    predict_model as predict_logistic_regression,
    predict_proba_model as predict_logistic_regression_proba,
    train_model as train_logistic_regression,
)
from src.models.mlp import (
    predict_model as predict_mlp,
    predict_proba_model as predict_mlp_proba,
    train_model as train_mlp,
)
from src.models.random_forest import (
    predict_model as predict_random_forest,
    predict_proba_model as predict_random_forest_proba,
    train_model as train_random_forest,
)


IDENTIFIER_COLUMNS = ("trip_id", "gps_observation_index", "candidate_arc_id")
TARGET_COLUMN = "is_match"
LABEL_STATUS_COLUMN = "label_status"
GROUND_TRUTH_COLUMNS = ("trip_id", "gps_observation_index", "label_arc_id")
METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
    "top1_accuracy",
    "training_seconds",
    "inference_seconds",
    "n_train_samples",
    "n_validation_samples",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    train: Callable[..., Pipeline]
    predict: Callable[..., np.ndarray]
    predict_proba: Callable[..., np.ndarray]
    predict_from_probability_threshold: bool = False


@dataclass
class PreparedDataset:
    X: pd.DataFrame
    y: np.ndarray
    metadata: pd.DataFrame
    labelled_observations: pd.DataFrame
    exclusions: dict[str, object]


@dataclass
class EvaluationResult:
    fold_results: pd.DataFrame
    comparison: pd.DataFrame
    exclusions: dict[str, object]
    best_model: str


def _model_specs() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            "Logistic Regression",
            train_logistic_regression,
            predict_logistic_regression,
            predict_logistic_regression_proba,
        ),
        ModelSpec(
            "Decision Tree",
            train_decision_tree,
            predict_decision_tree,
            predict_decision_tree_proba,
        ),
        ModelSpec(
            "Random Forest",
            train_random_forest,
            predict_random_forest,
            predict_random_forest_proba,
        ),
        ModelSpec("MLP", train_mlp, predict_mlp, predict_mlp_proba),
        ModelSpec(
            "Distance-weighted KNN",
            train_knn,
            predict_knn,
            predict_knn_proba,
            predict_from_probability_threshold=True,
        ),
    )


def load_training_data(
    candidate_features_path: str | Path,
    observations_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the processed training candidates and observation labels."""
    candidate_columns = [
        *IDENTIFIER_COLUMNS,
        *FEATURE_COLUMNS,
        "dataset_split",
        "label_arc_id",
        LABEL_STATUS_COLUMN,
        TARGET_COLUMN,
    ]
    observation_columns = [
        "trip_id",
        "gps_observation_index",
        "has_valid_coordinates",
        "label_arc_id",
        LABEL_STATUS_COLUMN,
        "dataset_split",
        "processing_status",
    ]
    candidates = pd.read_csv(
        candidate_features_path,
        usecols=candidate_columns,
        dtype={
            "trip_id": "string",
            "dataset_split": "string",
            LABEL_STATUS_COLUMN: "string",
        },
    )
    observations = pd.read_csv(
        observations_path,
        usecols=observation_columns,
        dtype={
            "trip_id": "string",
            "dataset_split": "string",
            LABEL_STATUS_COLUMN: "string",
            "processing_status": "string",
        },
    )
    return candidates, observations


def _require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...] | list[str],
    frame_name: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _require_training_split(frame: pd.DataFrame, frame_name: str) -> None:
    if "dataset_split" not in frame.columns:
        raise ValueError(f"{frame_name} is missing the dataset_split column.")
    splits = set(frame["dataset_split"].dropna().astype(str).unique())
    if splits != {"train"}:
        raise ValueError(
            f"{frame_name} must contain only the training split; found {sorted(splits)}."
        )


def prepare_evaluation_data(
    candidate_features: pd.DataFrame,
    observations: pd.DataFrame,
) -> PreparedDataset:
    """Select reliable labels and keep all non-feature metadata out of X."""
    _require_columns(
        candidate_features,
        [
            *IDENTIFIER_COLUMNS,
            *FEATURE_COLUMNS,
            "dataset_split",
            "label_arc_id",
            LABEL_STATUS_COLUMN,
            TARGET_COLUMN,
        ],
        "Candidate features",
    )
    _require_columns(
        observations,
        [
            *GROUND_TRUTH_COLUMNS,
            "has_valid_coordinates",
            LABEL_STATUS_COLUMN,
            "dataset_split",
            "processing_status",
        ],
        "Training observations",
    )
    _require_training_split(candidate_features, "Candidate features")
    _require_training_split(observations, "Training observations")

    observation_keys = ["trip_id", "gps_observation_index"]
    if observations.duplicated(observation_keys).any():
        raise ValueError("Training observations contain duplicate observation keys.")

    observation_labelled = observations[LABEL_STATUS_COLUMN].eq("labelled")
    valid_observation_mask = (
        observation_labelled
        & observations["processing_status"].eq("processed")
        & observations["has_valid_coordinates"].fillna(False).astype(bool)
        & observations["label_arc_id"].notna()
    )
    labelled_observations = observations.loc[
        valid_observation_mask, list(GROUND_TRUTH_COLUMNS)
    ].copy()
    for column in ("gps_observation_index", "label_arc_id"):
        numeric = pd.to_numeric(labelled_observations[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"Training observation column {column} must be integral.")
        labelled_observations[column] = numeric.astype(np.int64)
    labelled_observations["trip_id"] = labelled_observations["trip_id"].astype(str)

    status = candidate_features[LABEL_STATUS_COLUMN]
    target_is_binary = candidate_features[TARGET_COLUMN].isin((0, 1))
    candidate_labelled = status.eq("labelled")
    base_reliable_mask = candidate_labelled & target_is_binary
    reliable = candidate_features.loc[base_reliable_mask].copy()
    if reliable.empty:
        raise ValueError("No reliably labelled candidate rows are available.")

    for column in ("gps_observation_index", "candidate_arc_id", "label_arc_id"):
        numeric = pd.to_numeric(reliable[column], errors="raise")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"Candidate column {column} must contain integers.")
        reliable[column] = numeric.astype(np.int64)
    if reliable["trip_id"].isna().any():
        raise ValueError("Reliably labelled candidates contain missing trip_id values.")
    reliable["trip_id"] = reliable["trip_id"].astype(str)
    reliable[TARGET_COLUMN] = reliable[TARGET_COLUMN].astype(np.int64)

    target_matches_reference = reliable["candidate_arc_id"].eq(
        reliable["label_arc_id"]
    )
    if not np.array_equal(
        reliable[TARGET_COLUMN].to_numpy(dtype=np.int64),
        target_matches_reference.to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            "Candidate target values do not match candidate_arc_id and label_arc_id."
        )

    obs_truth = labelled_observations.rename(
        columns={"label_arc_id": "observation_label_arc_id"}
    )
    reliable = reliable.merge(
        obs_truth,
        on=observation_keys,
        how="left",
        validate="many_to_one",
    )
    mismatched_reference = (
        reliable["observation_label_arc_id"].notna()
        & reliable["label_arc_id"].ne(reliable["observation_label_arc_id"])
    )
    if mismatched_reference.any():
        raise ValueError(
            "Candidate and observation datasets disagree on labelled arc IDs."
        )
    has_observation_label = reliable["observation_label_arc_id"].notna()
    excluded_no_observation_label = int((~has_observation_label).sum())
    reliable = reliable.loc[has_observation_label].copy()
    if reliable.empty:
        raise ValueError("No candidate rows match valid training observation labels.")

    X = reliable.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in FEATURE_COLUMNS:
        X[column] = pd.to_numeric(X[column], errors="raise")
    missing_feature_values = int(X.isna().sum().sum())
    infinite_values = int(np.isinf(X.to_numpy(dtype=np.float64)).sum())
    if infinite_values:
        X = X.replace([np.inf, -np.inf], np.nan)

    y = reliable[TARGET_COLUMN].to_numpy(dtype=np.int64)
    metadata = reliable.loc[
        :,
        [
            "trip_id",
            "gps_observation_index",
            "candidate_arc_id",
            "label_arc_id",
        ],
    ].reset_index(drop=True)
    X = X.reset_index(drop=True)

    status_counts = (
        status.fillna("<missing>").astype(str).value_counts(dropna=False).to_dict()
    )
    invalid_target_count = int((candidate_labelled & ~target_is_binary).sum())
    excluded_by_status = {
        str(label_status): int(count)
        for label_status, count in status_counts.items()
        if label_status != "labelled"
    }

    positive_counts = (
        metadata.assign(_target=y)
        .groupby(observation_keys, sort=False)["_target"]
        .sum()
    )
    available_group_keys = positive_counts.loc[positive_counts.gt(0)].index
    available_group_keys = pd.DataFrame(
        list(available_group_keys), columns=observation_keys
    )
    multiple_positive_groups = int(positive_counts.gt(1).sum())
    observation_keys_available = set(
        map(
            tuple,
            available_group_keys.itertuples(index=False, name=None),
        )
    )
    gt_keys = list(
        map(
            tuple,
            labelled_observations[observation_keys].itertuples(
                index=False, name=None
            ),
        )
    )
    top1_available_observations = sum(key in observation_keys_available for key in gt_keys)

    labelled_observations = labelled_observations.reset_index(drop=True)
    exclusions: dict[str, object] = {
        "candidate_rows_total": int(len(candidate_features)),
        "candidate_rows_used": int(len(y)),
        "candidate_rows_excluded": int(len(candidate_features) - len(y)),
        "candidate_rows_excluded_by_label_status": excluded_by_status,
        "candidate_rows_labelled_with_invalid_target": invalid_target_count,
        "candidate_rows_labelled_without_valid_observation_label": (
            excluded_no_observation_label
        ),
        "missing_feature_values_before_pipeline_imputation": missing_feature_values,
        "candidate_rows_with_infinite_features_converted_to_missing": infinite_values,
        "source_training_trips": int(candidate_features["trip_id"].nunique()),
        "unique_training_trips": int(metadata["trip_id"].nunique()),
        "training_trips_without_reliable_candidates": int(
            candidate_features["trip_id"].nunique()
            - metadata["trip_id"].nunique()
        ),
        "labelled_observations_total": int(observation_labelled.sum()),
        "valid_ground_truth_observations": int(len(labelled_observations)),
        "ground_truth_observations_excluded_as_invalid_or_unprocessed": int(
            observation_labelled.sum() - len(labelled_observations)
        ),
        "top1_observations_with_available_correct_candidate": int(
            top1_available_observations
        ),
        "top1_observations_without_available_correct_candidate": int(
            len(labelled_observations) - top1_available_observations
        ),
        "observation_groups_with_multiple_positive_candidates": (
            multiple_positive_groups
        ),
        "positive_candidate_rows": int((y == 1).sum()),
        "negative_candidate_rows": int((y == 0).sum()),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    return PreparedDataset(
        X=X,
        y=y,
        metadata=metadata,
        labelled_observations=labelled_observations,
        exclusions=exclusions,
    )


def _classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    roc_auc = (
        float(roc_auc_score(y_true, probabilities))
        if np.unique(y_true).size == 2
        else float("nan")
    )
    probability_matrix = np.column_stack((1.0 - probabilities, probabilities))
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "log_loss": float(log_loss(y_true, probability_matrix, labels=[0, 1])),
    }


def _top1_metrics(
    metadata: pd.DataFrame,
    labelled_observations: pd.DataFrame,
    probabilities: np.ndarray,
    validation_trips: set[str],
    targets: np.ndarray,
) -> tuple[float, int, int, int, int]:
    keys = ["trip_id", "gps_observation_index"]
    truth = labelled_observations.loc[
        labelled_observations["trip_id"].isin(validation_trips),
        list(GROUND_TRUTH_COLUMNS),
    ]
    candidates = metadata.copy()
    candidates["target"] = targets
    candidates["probability"] = probabilities
    candidates = candidates.merge(
        truth,
        on=keys,
        how="inner",
        suffixes=("_candidate", "_observation"),
        validate="many_to_one",
    )
    if candidates.empty:
        return float("nan"), 0, len(truth), 0, 0

    candidate_label_column = "label_arc_id_candidate"
    truth_label_column = "label_arc_id_observation"
    if not candidates[candidate_label_column].eq(candidates[truth_label_column]).all():
        raise ValueError("Validation candidate labels disagree with observation labels.")

    correct_keys = candidates.loc[
        candidates["candidate_arc_id"].eq(candidates[truth_label_column]), keys
    ].drop_duplicates()
    available = candidates.merge(correct_keys, on=keys, how="inner")
    if available.empty:
        return float("nan"), 0, len(truth), 0, 0

    group_max = available.groupby(keys, sort=False)["probability"].transform("max")
    tied_winners = available.loc[available["probability"].eq(group_max)]
    tie_groups = int(
        tied_winners.groupby(keys, sort=False).size().gt(1).sum()
    )

    ranked = available.sort_values(
        [*keys, "probability", "candidate_arc_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    winners = ranked.drop_duplicates(keys, keep="first")
    top1_accuracy = float(
        winners["candidate_arc_id"].eq(winners[truth_label_column]).mean()
    )
    positive_counts = (
        available.loc[available["target"].eq(1)]
        .groupby(keys, sort=False)
        .size()
    )
    multiple_positive_groups = int(positive_counts.gt(1).sum())
    unavailable_observations = max(0, len(truth) - len(correct_keys))
    return (
        top1_accuracy,
        int(len(winners)),
        int(unavailable_observations),
        multiple_positive_groups,
        tie_groups,
    )


def _predict_validation_batch(
    spec: ModelSpec,
    model: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(
        spec.predict_proba(model, X, batch_size=max(1, len(X)))
        if spec.predict_from_probability_threshold
        else spec.predict_proba(model, X),
        dtype=np.float64,
    )
    if spec.predict_from_probability_threshold:
        # KNN's public predict_model uses this same threshold; reuse the
        # probabilities to avoid performing nearest-neighbor search twice.
        predictions = (probabilities >= 0.5).astype(np.int64)
    else:
        predictions = np.asarray(spec.predict(model, X), dtype=np.int64)
    if predictions.shape != (len(X),) or probabilities.shape != (len(X),):
        raise ValueError(f"{spec.name} returned unexpected prediction dimensions.")
    if not np.isin(predictions, (0, 1)).all():
        raise ValueError(f"{spec.name} returned predictions outside classes 0 and 1.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError(f"{spec.name} returned invalid class-1 probabilities.")
    return predictions, probabilities


def _aggregate_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in fold_results["model"].drop_duplicates():
        model_folds = fold_results.loc[fold_results["model"].eq(model_name)]
        row: dict[str, object] = {"model": model_name}
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(model_folds[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else (0.0 if len(values) == 1 else np.nan)
            )
        row["top1_observations_evaluated_total"] = int(
            model_folds["top1_observations_evaluated"].sum()
        )
        row["candidate_rows_evaluated_total"] = int(
            model_folds["n_validation_samples"].sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _select_best_model(comparison: pd.DataFrame) -> str:
    ranked = comparison.sort_values(
        [
            "top1_accuracy_mean",
            "log_loss_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "training_seconds_mean",
            "inference_seconds_mean",
        ],
        ascending=[False, True, False, False, False, True, True],
        na_position="last",
        kind="mergesort",
    )
    if ranked.empty or pd.isna(ranked.iloc[0]["top1_accuracy_mean"]):
        raise ValueError("No model produced an observation-level Top-1 score.")
    return str(ranked.iloc[0]["model"])


def _save_comparison_graphs(comparison: pd.DataFrame, figures_directory: Path) -> None:
    figures_directory.mkdir(parents=True, exist_ok=True)
    names = comparison["model"].tolist()
    x = np.arange(len(names))

    def save_bar(
        metric: str,
        title: str,
        y_label: str,
        filename: str,
        *,
        percentage: bool = False,
    ) -> None:
        means = comparison[f"{metric}_mean"].to_numpy(dtype=float)
        stds = comparison[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, means, yerr=stds, capsize=4, color="#4472C4")
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xticks(x, names, rotation=15, ha="right")
        if percentage:
            ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_directory / filename, dpi=160)
        plt.close(fig)

    save_bar(
        "accuracy",
        "Candidate-level Accuracy by Model",
        "Accuracy (mean ± standard deviation)",
        "accuracy_comparison.png",
        percentage=True,
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.24
    for offset, metric, label, color in (
        (-width, "precision", "Precision", "#70AD47"),
        (0.0, "recall", "Recall", "#ED7D31"),
        (width, "f1", "F1-score", "#4472C4"),
    ):
        ax.bar(
            x + offset,
            comparison[f"{metric}_mean"],
            width,
            yerr=comparison[f"{metric}_std"].fillna(0.0),
            capsize=3,
            label=label,
            color=color,
        )
    ax.set_title("Candidate-level Precision, Recall, and F1-score")
    ax.set_ylabel("Score (mean ± standard deviation)")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_directory / "precision_recall_f1_comparison.png", dpi=160)
    plt.close(fig)

    save_bar(
        "top1_accuracy",
        "Observation-level Top-1 Accuracy by Model",
        "Top-1 accuracy (mean ± standard deviation)",
        "top1_accuracy_comparison.png",
        percentage=True,
    )
    save_bar(
        "log_loss",
        "Candidate-level Log Loss by Model",
        "Log loss (mean ± standard deviation; lower is better)",
        "log_loss_comparison.png",
    )
    save_bar(
        "training_seconds",
        "Mean Model Training Time by Model",
        "Training time in seconds (mean ± standard deviation)",
        "training_time_comparison.png",
    )


def _metric_text(comparison_row: pd.Series, metric: str) -> str:
    return (
        f"{comparison_row[f'{metric}_mean']:.4f} ± "
        f"{comparison_row[f'{metric}_std']:.4f}"
    )


def _write_summary(
    summary_path: Path,
    comparison: pd.DataFrame,
    exclusions: dict[str, object],
    best_model: str,
    n_splits: int,
) -> None:
    best = comparison.loc[comparison["model"].eq(best_model)].iloc[0]
    sample_counts = comparison.iloc[0]
    features = exclusions["feature_columns"]
    excluded_statuses = exclusions["candidate_rows_excluded_by_label_status"]
    lines = [
        "# PathSnap Phase 5 Model Evaluation",
        "",
        "## Dataset and preparation",
        "",
        f"- Candidate rows in the prepared training artifact: {exclusions['candidate_rows_total']:,}.",
        f"- Candidate rows used for supervised evaluation: {exclusions['candidate_rows_used']:,} "
        f"({exclusions['positive_candidate_rows']:,} positive, "
        f"{exclusions['negative_candidate_rows']:,} negative).",
        f"- Candidate rows excluded: {exclusions['candidate_rows_excluded']:,}.",
        f"- Excluded by label status: `{excluded_statuses}`.",
        f"- Labelled candidates excluded for invalid targets: "
        f"{exclusions['candidate_rows_labelled_with_invalid_target']:,}.",
        f"- Labelled candidates excluded because no valid observation-level ground truth "
        f"was available: "
        f"{exclusions['candidate_rows_labelled_without_valid_observation_label']:,}.",
        f"- Source training trips: {exclusions['source_training_trips']:,}; "
        f"trips contributing reliable candidate rows: "
        f"{exclusions['unique_training_trips']:,}; trips with no reliable candidates: "
        f"{exclusions['training_trips_without_reliable_candidates']:,}.",
        f"- Candidate-level numeric feature columns ({len(features)}): "
        f"{', '.join(features)}.",
        f"- Missing values in the selected feature matrix before fold-local "
        f"imputation: {exclusions['missing_feature_values_before_pipeline_imputation']:,}; "
        f"infinite values converted to missing: "
        f"{exclusions['candidate_rows_with_infinite_features_converted_to_missing']:,}.",
        "- `trip_id`, `gps_observation_index`, and `candidate_arc_id` were retained as "
        "metadata and excluded from `X`; label/reference columns were not features.",
        "- Only the prepared training candidate and observation artifacts were read. "
        "Student-test and evaluator data were not used.",
        "",
        "## Validation procedure",
        "",
        f"- {n_splits}-fold `GroupKFold`, grouped by `trip_id`; each fold uses the same "
        "candidate rows and held-out trips for all five models. All recorded "
        "train/validation trip-overlap checks were zero.",
        f"- Mean candidate sample counts per fold: "
        f"{sample_counts['n_train_samples_mean']:,.0f} training and "
        f"{sample_counts['n_validation_samples_mean']:,.0f} validation.",
        "- Imputation and any model-specific scaling are fit inside each model "
        "pipeline on that fold's training rows only; tree-based models are unscaled.",
        "- Candidate-level recall is positive-class classifier recall, not Phase 3 "
        "candidate recall.",
        "- For Top-1, only confidently labelled observations with an available correct "
        "candidate are counted. Equal-probability ties are resolved by the smallest "
        "`candidate_arc_id`; any positive candidate is considered correct.",
        f"- Valid ground-truth observations: {exclusions['valid_ground_truth_observations']:,}; "
        f"with a correct candidate available: "
        f"{exclusions['top1_observations_with_available_correct_candidate']:,}; "
        f"without one: "
        f"{exclusions['top1_observations_without_available_correct_candidate']:,}.",
        f"- Observation groups with multiple positive candidate rows: "
        f"{exclusions['observation_groups_with_multiple_positive_candidates']:,}.",
        "",
        "## Five-fold results",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss | Top-1 accuracy | Training time (s) | Inference time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['model']} | {_metric_text(row, 'accuracy')} | "
            f"{_metric_text(row, 'precision')} | {_metric_text(row, 'recall')} | "
            f"{_metric_text(row, 'f1')} | {_metric_text(row, 'roc_auc')} | "
            f"{_metric_text(row, 'log_loss')} | "
            f"{_metric_text(row, 'top1_accuracy')} | "
            f"{_metric_text(row, 'training_seconds')} | "
            f"{_metric_text(row, 'inference_seconds')} |"
        )
    lines.extend(
        [
            "",
            "All metric and timing entries above are mean ± standard deviation across "
            f"the {n_splits} held-out-trip folds. Fold-wise values and sample counts are "
            "saved in `model_fold_results.csv`; aggregate values are in "
            "`model_comparison.csv`.",
            "",
            "## Best model and trade-offs",
            "",
            f"**Selected model: {best_model}.** Selection follows the stated priority: "
            "highest mean observation-level Top-1 accuracy, then lower mean log loss, "
            "then candidate precision, recall, F1, and training/inference time. "
            "Its measured mean "
            f"Top-1 accuracy was {_metric_text(best, 'top1_accuracy')}; mean log loss "
            f"was {_metric_text(best, 'log_loss')}.",
            "",
            "This is a grouped cross-validation comparison on the current training trips, "
            "not an external test-set result. The folds share one finite set of trips, so "
            "the reported standard deviations describe fold variation and are not "
            "independent confidence intervals.",
            "",
            "Comparison figures are saved under `figures/`.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_models(
    candidate_features: pd.DataFrame,
    observations: pd.DataFrame,
    output_directory: str | Path,
    *,
    n_splits: int = 5,
    inference_batch_size: int = 8192,
) -> EvaluationResult:
    """Train and evaluate the five existing models using grouped CV."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive.")
    prepared = prepare_evaluation_data(candidate_features, observations)
    if prepared.metadata["trip_id"].nunique() < n_splits:
        raise ValueError(
            f"GroupKFold with {n_splits} folds requires at least {n_splits} trips."
        )

    specs = _model_specs()
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(prepared.X, prepared.y, prepared.metadata["trip_id"]))
    fold_rows: list[dict[str, object]] = []

    for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_trips = set(prepared.metadata.iloc[train_indices]["trip_id"])
        validation_trips = set(prepared.metadata.iloc[validation_indices]["trip_id"])
        overlap = train_trips.intersection(validation_trips)
        if overlap:
            raise RuntimeError(f"Trip leakage detected in fold {fold_number}: {overlap}")
        validation_trip_string = "|".join(sorted(validation_trips))
        X_train = prepared.X.iloc[train_indices]
        y_train = prepared.y[train_indices]
        X_validation = prepared.X.iloc[validation_indices]
        y_validation = prepared.y[validation_indices]
        validation_metadata = prepared.metadata.iloc[validation_indices].reset_index(
            drop=True
        )

        for spec in specs:
            start = perf_counter()
            model = spec.train(X_train, y_train)
            training_seconds = perf_counter() - start

            prediction_parts: list[np.ndarray] = []
            probability_parts: list[np.ndarray] = []
            start = perf_counter()
            if spec.predict_from_probability_threshold:
                predictions, probabilities = _predict_validation_batch(
                    spec, model, X_validation
                )
                prediction_parts.append(predictions)
                probability_parts.append(probabilities)
            else:
                for batch_start in range(0, len(X_validation), inference_batch_size):
                    batch_end = min(
                        batch_start + inference_batch_size, len(X_validation)
                    )
                    X_batch = X_validation.iloc[batch_start:batch_end]
                    predictions, probabilities = _predict_validation_batch(
                        spec, model, X_batch
                    )
                    prediction_parts.append(predictions)
                    probability_parts.append(probabilities)
            inference_seconds = perf_counter() - start
            predictions = np.concatenate(prediction_parts)
            probabilities = np.concatenate(probability_parts)
            metrics = _classification_metrics(
                y_validation, predictions, probabilities
            )
            (
                top1_accuracy,
                top1_observations,
                top1_unavailable,
                multiple_positive_groups,
                tie_groups,
            ) = _top1_metrics(
                validation_metadata,
                prepared.labelled_observations,
                probabilities,
                validation_trips,
                y_validation,
            )
            fold_rows.append(
                {
                    "fold": fold_number,
                    "model": spec.name,
                    **metrics,
                    "top1_accuracy": top1_accuracy,
                    "top1_observations_evaluated": top1_observations,
                    "top1_observations_without_available_correct_candidate": (
                        top1_unavailable
                    ),
                    "top1_multiple_positive_groups": multiple_positive_groups,
                    "top1_probability_tie_groups": tie_groups,
                    "training_seconds": training_seconds,
                    "inference_seconds": inference_seconds,
                    "n_train_samples": int(len(train_indices)),
                    "n_validation_samples": int(len(validation_indices)),
                    "n_train_trips": len(train_trips),
                    "n_validation_trips": len(validation_trips),
                    "trip_overlap_count": len(overlap),
                    "validation_trip_ids": validation_trip_string,
                }
            )

    fold_results = pd.DataFrame(fold_rows)
    comparison = _aggregate_results(fold_results)
    best_model = _select_best_model(comparison)
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(out / "model_fold_results.csv", index=False)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    _save_comparison_graphs(comparison, out / "figures")
    _write_summary(
        out / "model_evaluation_summary.md",
        comparison,
        prepared.exclusions,
        best_model,
        n_splits,
    )
    return EvaluationResult(
        fold_results=fold_results,
        comparison=comparison,
        exclusions=prepared.exclusions,
        best_model=best_model,
    )


def run_phase5_evaluation(
    project_root: str | Path = ".",
    *,
    n_splits: int = 5,
) -> EvaluationResult:
    """Load only the prepared training artifacts and run Phase 5 evaluation."""
    root = Path(project_root)
    candidates, observations = load_training_data(
        root / "data" / "processed" / "train_candidate_features.csv",
        root / "data" / "processed" / "train_observations.csv",
    )
    return evaluate_models(
        candidates,
        observations,
        root / "reports",
        n_splits=n_splits,
    )


def main() -> None:
    result = run_phase5_evaluation(Path(__file__).resolve().parents[1])
    print(result.comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected model: {result.best_model}")


if __name__ == "__main__":
    main()
