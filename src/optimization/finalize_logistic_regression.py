"""Fit, export, and evaluate the selected Phase 6 Logistic Regression."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.candidates import generate_candidates
from src.evaluation import build_route_reference_candidates
from src.features import FEATURE_COLUMNS, extract_candidate_features
from src.io import load_arcs, load_nodes, load_route, load_track
from src.labeling import align_route_labels
from src.model_evaluation import (
    _classification_metrics,
    _top1_metrics,
    load_training_data,
    prepare_evaluation_data,
)
from src.optimization.optimized_logistic_regression import (
    load_optimized_parameters,
    predict_optimized_proba,
    train_optimized_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_RADIUS_M = 50.0
MAX_CANDIDATES = 8
AMBIGUITY_MARGIN_M = 5.0
CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")
COMPARISON_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
    "top1_accuracy",
    "training_seconds",
    "inference_seconds",
)
GROUND_TRUTH_COLUMNS = ("trip_id", "gps_observation_index", "label_arc_id")
OBSERVATION_KEYS = ("trip_id", "gps_observation_index")


def _empty_classification_metrics() -> dict[str, float]:
    return {metric: float("nan") for metric in CLASSIFICATION_METRICS}


def score_predictions(
    metadata: pd.DataFrame,
    observation_labels: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float | int]:
    """Score row-aligned candidate predictions against route-aligned labels."""
    prediction_values = np.asarray(predictions)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    target_values = np.asarray(targets)
    row_count = len(metadata)
    if (
        prediction_values.shape != (row_count,)
        or probability_values.shape != (row_count,)
        or target_values.shape != (row_count,)
    ):
        raise ValueError("Candidate metadata, targets, predictions, and probabilities must align.")
    if not np.isin(prediction_values, (0, 1)).all():
        raise ValueError("Predictions must contain only binary classes 0 and 1.")
    if not np.isin(target_values, (0, 1)).all():
        raise ValueError("Candidate targets must contain only binary labels 0 and 1.")
    if not np.isfinite(probability_values).all() or (
        (probability_values < 0.0) | (probability_values > 1.0)
    ).any():
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")

    required_metadata = {
        *OBSERVATION_KEYS,
        "candidate_arc_id",
        "label_arc_id",
    }
    if required_metadata.difference(metadata.columns):
        raise ValueError("Candidate metadata is missing required identifiers or labels.")
    if set(GROUND_TRUTH_COLUMNS).difference(observation_labels.columns):
        raise ValueError("Observation labels are missing required ground-truth columns.")

    truth = observation_labels.loc[:, list(GROUND_TRUTH_COLUMNS)].copy()
    if row_count:
        classification = _classification_metrics(
            target_values.astype(np.int64),
            prediction_values.astype(np.int64),
            probability_values,
        )
    else:
        classification = _empty_classification_metrics()

    top1, evaluated_count, missed_count, _, _ = _top1_metrics(
        metadata.reset_index(drop=True),
        truth,
        probability_values,
        set(truth["trip_id"].astype(str)),
        target_values.astype(np.int64),
    )
    result: dict[str, float | int] = dict(classification)
    result.update(
        {
            "top1_accuracy": top1,
            "top1_observations_evaluated": evaluated_count,
            "labelled_observations": int(len(truth)),
            "candidate_miss_count": missed_count,
            "candidate_miss_rate": (
                float(missed_count / len(truth)) if len(truth) else float("nan")
            ),
            "candidate_rows_evaluated": int(row_count),
        }
    )
    return result


def _evaluate_student_trip(
    trip_directory: Path,
    model: Any,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    trip_id = trip_directory.name
    track = load_track(trip_directory / f"{trip_id}.track")
    nodes = load_nodes(trip_directory / f"{trip_id}.nodes")
    arcs = load_arcs(trip_directory / f"{trip_id}.arcs")
    route = load_route(trip_directory / f"{trip_id}.route")

    reference_candidates = build_route_reference_candidates(
        track,
        nodes,
        arcs,
        route,
        trip_id,
        reference_radius_m=SEARCH_RADIUS_M,
    )
    observation_labels = align_route_labels(
        route,
        arcs,
        reference_candidates,
        len(track),
        trip_id,
        ambiguity_margin_m=AMBIGUITY_MARGIN_M,
    )
    generated = generate_candidates(
        track,
        nodes,
        arcs,
        trip_id,
        search_radius_m=SEARCH_RADIUS_M,
        max_candidates=MAX_CANDIDATES,
    )
    feature_rows = extract_candidate_features(
        track, nodes, arcs, generated.candidates, trip_id
    )
    candidate_rows = feature_rows.merge(
        observation_labels,
        on=list(OBSERVATION_KEYS),
        how="left",
        validate="many_to_one",
    )
    labelled_rows = candidate_rows.loc[
        candidate_rows["label_status"].eq("labelled")
        & candidate_rows["label_arc_id"].notna()
    ].copy()
    truth = observation_labels.loc[
        observation_labels["label_status"].eq("labelled"),
        list(GROUND_TRUTH_COLUMNS),
    ].copy()
    truth["trip_id"] = truth["trip_id"].astype(str)
    truth["gps_observation_index"] = truth["gps_observation_index"].astype(np.int64)
    truth["label_arc_id"] = truth["label_arc_id"].astype(np.int64)

    metadata = labelled_rows.loc[
        :,
        [
            "trip_id",
            "gps_observation_index",
            "candidate_arc_id",
            "label_arc_id",
        ],
    ].copy()
    targets = metadata["candidate_arc_id"].eq(metadata["label_arc_id"]).to_numpy(
        dtype=np.int64
    )
    X = labelled_rows.loc[:, list(FEATURE_COLUMNS)].copy()
    inference_start = perf_counter()
    if len(X):
        predictions = np.asarray(model.predict(X), dtype=np.int64)
        probabilities = predict_optimized_proba(model, X)
    else:
        predictions = np.empty(0, dtype=np.int64)
        probabilities = np.empty(0, dtype=np.float64)
    inference_seconds = perf_counter() - inference_start
    metrics = score_predictions(metadata, truth, predictions, probabilities, targets)
    status = "evaluated" if len(truth) else "evaluated_no_labelled_observations"
    row: dict[str, Any] = {
        "trip_id": trip_id,
        "status": status,
        "error_reason": "",
        "observation_count": int(len(track)),
        "labelled_observations": metrics["labelled_observations"],
        "candidate_rows_generated": int(len(feature_rows)),
        "candidate_rows_evaluated": metrics["candidate_rows_evaluated"],
        "inference_seconds": inference_seconds,
        **{metric: metrics[metric] for metric in (*CLASSIFICATION_METRICS, "top1_accuracy")},
        "candidate_miss_count": metrics["candidate_miss_count"],
        "candidate_miss_rate": metrics["candidate_miss_rate"],
        "top1_observations_evaluated": metrics["top1_observations_evaluated"],
    }
    return row, metadata, truth, targets, predictions, probabilities


def evaluate_student_test(
    student_test_directory: str | Path,
    model: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate once on student-test; route files are used only for scoring."""
    split_directory = Path(student_test_directory)
    if not split_directory.is_dir():
        raise FileNotFoundError(f"Student-test directory was not found: {split_directory}")
    trip_directories = sorted(
        (path for path in split_directory.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not trip_directories:
        raise ValueError(f"No student-test trip directories found in {split_directory}")

    rows: list[dict[str, Any]] = []
    metadata_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    targets_list: list[np.ndarray] = []
    predictions_list: list[np.ndarray] = []
    probabilities_list: list[np.ndarray] = []
    for trip_directory in trip_directories:
        try:
            row, metadata, truth, targets, predictions, probabilities = (
                _evaluate_student_trip(trip_directory, model)
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.append(
                {
                    "trip_id": trip_directory.name,
                    "status": "failed",
                    "error_reason": f"{type(error).__name__}: {error}",
                    "observation_count": float("nan"),
                    "labelled_observations": 0,
                    "candidate_rows_generated": 0,
                    "candidate_rows_evaluated": 0,
                    "inference_seconds": 0.0,
                    **{metric: float("nan") for metric in CLASSIFICATION_METRICS},
                    "top1_accuracy": float("nan"),
                    "candidate_miss_count": 0,
                    "candidate_miss_rate": float("nan"),
                    "top1_observations_evaluated": 0,
                }
            )
            continue
        rows.append(row)
        metadata_frames.append(metadata)
        truth_frames.append(truth)
        targets_list.append(targets)
        predictions_list.append(predictions)
        probabilities_list.append(probabilities)

    per_trip = pd.DataFrame(rows)
    if not truth_frames:
        aggregate: dict[str, Any] = {
            **_empty_classification_metrics(),
            "top1_accuracy": float("nan"),
            "top1_observations_evaluated": 0,
            "labelled_observations": 0,
            "candidate_miss_count": 0,
            "candidate_miss_rate": float("nan"),
            "candidate_rows_evaluated": 0,
            "inference_seconds": 0.0,
        }
    else:
        metadata = pd.concat(metadata_frames, ignore_index=True)
        truth = pd.concat(truth_frames, ignore_index=True)
        targets = np.concatenate(targets_list)
        predictions = np.concatenate(predictions_list)
        probabilities = np.concatenate(probabilities_list)
        aggregate = score_predictions(
            metadata, truth, predictions, probabilities, targets
        )
        aggregate["inference_seconds"] = float(
            per_trip.loc[per_trip["status"].ne("failed"), "inference_seconds"].sum()
        )
    aggregate["trips_found"] = int(len(trip_directories))
    aggregate["trips_evaluated"] = int(per_trip["status"].str.startswith("evaluated").sum())
    aggregate["trips_failed"] = int(per_trip["status"].eq("failed").sum())
    aggregate["candidate_rows_generated"] = int(per_trip["candidate_rows_generated"].sum())
    return per_trip, aggregate


def _historical_comparison_rows(
    project_root: Path,
    training_trip_count: int,
) -> list[dict[str, Any]]:
    baseline_path = project_root / "reports" / "model_comparison.csv"
    nested_path = project_root / "reports" / "optimization" / "nested_cv_results.csv"
    baseline = pd.read_csv(baseline_path)
    baseline_rows = baseline.loc[baseline["model"].eq("Logistic Regression")]
    if len(baseline_rows) != 1:
        raise ValueError("Phase 5 comparison must contain one Logistic Regression row.")
    baseline_values = baseline_rows.iloc[0]
    rows: list[dict[str, Any]] = []
    baseline_row: dict[str, Any] = {
        "model": "Logistic Regression",
        "model_configuration": "C=1.0, max_iter=1000, solver=lbfgs, random_state=42",
        "evaluation_method": "Phase 5 five-fold GroupKFold",
        "training_trips": training_trip_count,
        "evaluated_trips": training_trip_count,
        "candidate_miss_rate": np.nan,
        "notes": "Untuned baseline; historical CV result, not a student-test score.",
    }
    for metric in COMPARISON_METRICS:
        baseline_row[metric] = float(baseline_values[f"{metric}_mean"])
        baseline_row[f"{metric}_std"] = float(baseline_values[f"{metric}_std"])
    rows.append(baseline_row)

    nested = pd.read_csv(nested_path)
    if nested.empty:
        raise ValueError("Phase 6 nested-CV results are empty.")
    nested_row = {
        "model": "Logistic Regression",
        "model_configuration": (
            "Fold-specific Optuna C; final selected C="
            f"{load_optimized_parameters()['C']:.12g}"
        ),
        "evaluation_method": "Phase 6 five-fold nested GroupKFold",
        "training_trips": training_trip_count,
        "evaluated_trips": int(nested["n_validation_trips"].sum()),
        "candidate_miss_rate": np.nan,
        "notes": (
            "Outer-fold tuned results; fold-specific C values were selected by "
            "inner GroupKFold. Not directly comparable to Phase 5 or student-test."
        ),
    }
    for metric in COMPARISON_METRICS:
        values = pd.to_numeric(nested[metric], errors="raise")
        nested_row[metric] = float(values.mean())
        nested_row[f"{metric}_std"] = float(values.std(ddof=1))
    rows.append(nested_row)
    return rows


def _write_summary(
    output_path: Path,
    parameters: dict[str, Any],
    prepared_rows: int,
    training_trips: int,
    source_training_trips: int,
    training_seconds: float,
    artifact_path: Path,
    artifact_size: int,
    student_metrics: dict[str, Any] | None,
    per_trip: pd.DataFrame | None,
    comparison: pd.DataFrame | None,
) -> None:
    lines = [
        "# Phase 6 Final Tuned Logistic Regression",
        "",
        "## Final refit and artifact",
        "",
        f"- Selected parameters: `{parameters}`.",
        (
            f"- Fit on {prepared_rows:,} reliable candidate rows from "
            f"{training_trips} eligible trips ({source_training_trips} source training trips)."
        ),
        "- Only the existing training split was used for fitting; preprocessing remains inside the fitted pipeline.",
        f"- Training time: {training_seconds:.3f} seconds.",
        f"- Artifact: `{artifact_path}` ({artifact_size:,} bytes), saved and loaded successfully with joblib.",
        "- The loaded pipeline preserved class order `[0, 1]` and produced finite, row-aligned probabilities.",
        "",
        "## Student-test evaluation",
        "",
        "Student-test route files were used only to create route-aligned evaluation labels; they were not used for fitting, feature generation, threshold selection, or model selection.",
    ]
    if student_metrics is None or per_trip is None:
        lines.extend(
            [
                "",
                "Student-test evaluation was skipped (`--artifact-only`).",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"- Trips found: {student_metrics['trips_found']}; evaluated: {student_metrics['trips_evaluated']}; failed: {student_metrics['trips_failed']}.",
                f"- Candidate rows evaluated: {student_metrics['candidate_rows_evaluated']:,}; labelled observations: {student_metrics['labelled_observations']:,}.",
                "",
                "| Metric | Result |",
                "|---|---:|",
            ]
        )
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "top1_accuracy",
            "candidate_miss_rate",
            "training_seconds",
            "inference_seconds",
        ):
            value = (
                training_seconds
                if metric == "training_seconds"
                else student_metrics.get(metric, float("nan"))
            )
            lines.append(f"| {metric} | {value:.6f} |")
        failed = per_trip.loc[per_trip["status"].eq("failed")]
        if not failed.empty:
            lines.extend(
                [
                    "",
                    "Trips not evaluated (with reasons):",
                    *[
                        f"- `{row.trip_id}`: {row.error_reason}"
                        for row in failed.itertuples(index=False)
                    ],
                ]
            )
    lines.extend(
        [
            "",
            "## Tuned-versus-untuned context",
            "",
            "The comparison CSV keeps three protocols distinct: Phase 5 untuned GroupKFold, Phase 6 nested GroupKFold, and the final tuned model on student-test.",
        ]
    )
    if comparison is not None:
        lines.extend(
            [
                "",
                "| Evaluation protocol | Configuration | Top-1 accuracy | Log loss | Training (s) | Inference (s) | Candidate miss rate |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in comparison.itertuples(index=False):
            top1 = f"{row.top1_accuracy:.6f}"
            if np.isfinite(row.top1_accuracy_std):
                top1 += f" ± {row.top1_accuracy_std:.6f}"
            log_loss = f"{row.log_loss:.6f}"
            if np.isfinite(row.log_loss_std):
                log_loss += f" ± {row.log_loss_std:.6f}"
            miss_rate = (
                f"{row.candidate_miss_rate:.6f}"
                if np.isfinite(row.candidate_miss_rate)
                else "n/a"
            )
            lines.append(
                f"| {row.evaluation_method} | {row.model_configuration} | "
                f"{top1} | {log_loss} | {row.training_seconds:.3f} | "
                f"{row.inference_seconds:.3f} | {miss_rate} |"
            )
        baseline_rows = comparison.loc[
            comparison["evaluation_method"].eq("Phase 5 five-fold GroupKFold")
        ]
        nested_rows = comparison.loc[
            comparison["evaluation_method"].eq("Phase 6 five-fold nested GroupKFold")
        ]
        if len(baseline_rows) == 1 and len(nested_rows) == 1:
            baseline_row = baseline_rows.iloc[0]
            nested_row = nested_rows.iloc[0]
            top1_delta = nested_row["top1_accuracy"] - baseline_row["top1_accuracy"]
            log_loss_delta = nested_row["log_loss"] - baseline_row["log_loss"]
            lines.extend(
                [
                    "",
                    (
                        f"- Phase 6 nested tuned Top-1 differed from the Phase 5 "
                        f"untuned baseline by {top1_delta:+.6f}; log loss differed "
                        f"by {log_loss_delta:+.6f} (negative is lower). These are "
                        "different validation protocols, so the differences do not "
                        "establish a tuning improvement."
                    ),
                ]
            )
    else:
        lines.extend(["", "Historical comparison was not regenerated (`--artifact-only`)."])
    lines.extend(
        [
            "",
            "The student-test C=1 baseline was not refit, per the authorized tuned-only scope; therefore there is no matched tuned-versus-untuned student-test comparison and no claim of improvement from cross-protocol results.",
            "",
            "Observation-level Top-1 accuracy remains the primary selection metric. Log loss is reported as a secondary probability-sensitive metric because downstream map matching consumes classifier probabilities; lower log loss penalizes confident errors.",
            "",
            "## Reproduction",
            "",
            "From the repository root, run:",
            "",
            "```powershell",
            "python -m src.optimization.finalize_logistic_regression",
            "```",
            "",
            "To regenerate only the model artifact without reading student-test data:",
            "",
            "```powershell",
            "python -m src.optimization.finalize_logistic_regression --artifact-only",
            "```",
            "",
            f"API model path: set `PATHSNAP_MODEL_PATH` to `{artifact_path}`. The API configuration is not changed by this workflow.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_finalization(
    data_root: str | Path = PROJECT_ROOT / "data",
    artifact_path: str | Path = PROJECT_ROOT / "artifacts" / "model.joblib",
    reports_directory: str | Path = PROJECT_ROOT / "reports" / "optimization",
    *,
    artifact_only: bool = False,
) -> dict[str, Any]:
    data_path = Path(data_root)
    artifact = Path(artifact_path)
    reports = Path(reports_directory)
    candidates, observations = load_training_data(
        data_path / "processed" / "train_candidate_features.csv",
        data_path / "processed" / "train_observations.csv",
    )
    prepared = prepare_evaluation_data(candidates, observations)
    parameters = load_optimized_parameters()

    training_start = perf_counter()
    model = train_optimized_model(prepared.X, prepared.y)
    training_seconds = perf_counter() - training_start
    classifier = model.named_steps["logistic_regression"]
    if not np.array_equal(classifier.classes_, np.array([0, 1])):
        raise ValueError(f"Final model classes must be [0, 1], got {classifier.classes_}.")

    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact, compress=3)
    loaded_model = joblib.load(artifact)
    check_rows = prepared.X.iloc[: min(128, len(prepared.X))]
    check_probabilities = predict_optimized_proba(loaded_model, check_rows)
    if check_probabilities.shape != (len(check_rows),):
        raise ValueError("Loaded artifact returned a probability count mismatch.")
    if not np.isfinite(check_probabilities).all() or (
        (check_probabilities < 0.0) | (check_probabilities > 1.0)
    ).any():
        raise ValueError("Loaded artifact returned invalid probabilities.")
    if not np.array_equal(
        np.asarray(loaded_model.predict(check_rows)),
        np.asarray(model.predict(check_rows)),
    ):
        raise ValueError("Loaded artifact predictions differ from the fitted pipeline.")

    per_trip: pd.DataFrame | None = None
    student_metrics: dict[str, Any] | None = None
    comparison: pd.DataFrame | None = None
    if not artifact_only:
        per_trip, student_metrics = evaluate_student_test(
            data_path / "student_test",
            loaded_model,
        )
        student_metrics["training_seconds"] = training_seconds
        student_metrics["model_configuration"] = (
            f"Tuned final refit, C={parameters['C']:.12g}"
        )
        reports.mkdir(parents=True, exist_ok=True)
        per_trip.to_csv(reports / "student_test_per_trip.csv", index=False)
        pd.DataFrame([student_metrics]).to_csv(
            reports / "student_test_results.csv", index=False
        )

        comparison_rows = _historical_comparison_rows(
            PROJECT_ROOT,
            int(prepared.metadata["trip_id"].nunique()),
        )
        student_row: dict[str, Any] = {
            "model": "Logistic Regression",
            "model_configuration": f"Tuned final refit, C={parameters['C']:.12g}",
            "evaluation_method": "Final refit; student-test, one evaluation",
            "training_trips": int(prepared.metadata["trip_id"].nunique()),
            "evaluated_trips": student_metrics["trips_evaluated"],
            "candidate_miss_rate": student_metrics["candidate_miss_rate"],
            "notes": (
                "Only the tuned model was evaluated on student-test; no untuned "
                "student-test refit was authorized."
            ),
        }
        for metric in COMPARISON_METRICS:
            student_row[metric] = (
                training_seconds
                if metric == "training_seconds"
                else student_metrics.get(metric, np.nan)
            )
            student_row[f"{metric}_std"] = np.nan
        comparison_rows.append(student_row)
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(
            reports / "tuned_vs_untuned_comparison.csv",
            index=False,
        )

    summary_path = reports / "final_refit_summary.md"
    _write_summary(
        summary_path,
        parameters,
        len(prepared.y),
        int(prepared.metadata["trip_id"].nunique()),
        int(prepared.exclusions["source_training_trips"]),
        training_seconds,
        artifact,
        artifact.stat().st_size,
        student_metrics,
        per_trip,
        comparison,
    )
    if student_metrics is not None and student_metrics["trips_evaluated"] == 0:
        raise RuntimeError(
            "No student-test trips could be scored; see student_test_per_trip.csv."
        )
    return {
        "model": loaded_model,
        "parameters": parameters,
        "training_rows": len(prepared.y),
        "training_trips": int(prepared.metadata["trip_id"].nunique()),
        "training_seconds": training_seconds,
        "artifact_path": artifact,
        "artifact_size": artifact.stat().st_size,
        "student_test": student_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "model.joblib",
    )
    parser.add_argument(
        "--reports-directory",
        type=Path,
        default=PROJECT_ROOT / "reports" / "optimization",
    )
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="Fit and export from training data without opening student-test trips.",
    )
    args = parser.parse_args()
    result = run_finalization(
        args.data_root,
        args.artifact_path,
        args.reports_directory,
        artifact_only=args.artifact_only,
    )
    print(
        f"Saved {result['artifact_path']} ({result['artifact_size']:,} bytes); "
        f"trained on {result['training_rows']:,} candidate rows."
    )
    if result["student_test"] is not None:
        print(
            "Student-test: "
            f"Top-1={result['student_test']['top1_accuracy']:.6f}, "
            f"log_loss={result['student_test']['log_loss']:.6f}, "
            f"trips={result['student_test']['trips_evaluated']}/"
            f"{result['student_test']['trips_found']}."
        )


if __name__ == "__main__":
    main()
