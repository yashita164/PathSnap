"""End-to-end Phase 3 candidate and training-label processing."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates import generate_candidates
from src.evaluation import build_route_reference_candidates, candidate_recall_summary
from src.features import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    extract_candidate_features,
)
from src.io import load_arcs, load_nodes, load_route, load_track
from src.labeling import align_route_labels, split_route_components


DEFAULT_RADII_M = (25.0, 50.0, 75.0, 100.0)


def _trip_directories(split_directory: Path) -> list[Path]:
    """Find trip directories once, preserving stable path order."""
    trip_directories = {
        path.parent
        for path in split_directory.rglob("*.track")
    }
    return sorted(trip_directories)


def _issue(
    trip_id: str,
    dataset_split: str,
    issue_type: str,
    description: str,
) -> dict[str, str]:
    return {
        "trip_id": trip_id,
        "dataset_split": dataset_split,
        "issue_type": issue_type,
        "description": description,
    }


def _empty_observation_rows(
    track: pd.DataFrame,
    trip_id: str,
    dataset_split: str,
    processing_status: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": str(trip_id),
            "dataset_split": dataset_split,
            "gps_observation_index": np.arange(len(track), dtype=np.int64),
            "has_valid_coordinates": (
                np.isfinite(pd.to_numeric(track["longitude"], errors="coerce"))
                & np.isfinite(pd.to_numeric(track["latitude"], errors="coerce"))
            ),
            "candidate_count": np.zeros(len(track), dtype=np.int64),
            "nearby_arc_count": np.zeros(len(track), dtype=np.int64),
            "search_radius_m": np.nan,
            "processing_status": processing_status,
        }
    )


def _weighted_recall_summary(
    trip_summaries: list[pd.DataFrame],
    radius_values: tuple[float, ...],
) -> pd.DataFrame:
    if not trip_summaries:
        return pd.DataFrame(
            columns=(
                "search_radius_m",
                "reference_label_count",
                "candidate_hits",
                "candidate_recall",
                "label_coverage",
                "observations_with_candidates",
                "observations_without_candidates",
            )
        )
    combined = pd.concat(trip_summaries, ignore_index=True)
    rows = []
    for radius_m in radius_values:
        subset = combined.loc[combined["search_radius_m"] == radius_m]
        label_count = int(subset["reference_label_count"].sum())
        hits = int(subset["candidate_hits"].sum())
        gps_observations = int(subset["gps_observation_count"].sum())
        rows.append(
            {
                "search_radius_m": radius_m,
                "reference_label_count": label_count,
                "candidate_hits": hits,
                "candidate_recall": hits / label_count if label_count else np.nan,
                "label_coverage": (
                    label_count / gps_observations if gps_observations else np.nan
                ),
                "observations_with_candidates": int(
                    subset["observations_with_candidates"].sum()
                ),
                "observations_without_candidates": int(
                    subset["observations_without_candidates"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_phase3(
    data_root: str | Path = "data",
    output_directory: str | Path = "data/processed",
    search_radius_m: float = 50.0,
    max_candidates: int = 8,
    ambiguity_margin_m: float = 5.0,
    evaluation_radii_m: tuple[float, ...] = DEFAULT_RADII_M,
) -> dict[str, pd.DataFrame]:
    """Process training and student-test trips into separate CSV artifacts.

    Student-test route files are never opened. Training route files are used
    exclusively to establish labels and evaluate candidate recall.
    """
    if not evaluation_radii_m or any(
        not np.isfinite(radius) or radius <= 0 for radius in evaluation_radii_m
    ):
        raise ValueError("evaluation_radii_m must contain positive finite radii")
    if search_radius_m not in evaluation_radii_m:
        evaluation_radii_m = tuple(sorted((*evaluation_radii_m, search_radius_m)))

    root = Path(data_root)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    feature_columns = [
        *IDENTIFIER_COLUMNS,
        *FEATURE_COLUMNS,
        "dataset_split",
    ]
    train_features: list[pd.DataFrame] = []
    student_features: list[pd.DataFrame] = []
    train_observations: list[pd.DataFrame] = []
    student_observations: list[pd.DataFrame] = []
    trip_evaluation_summaries: list[pd.DataFrame] = []
    issues: list[dict[str, str]] = []
    route_summaries: list[dict[str, int | str]] = []

    for dataset_split in ("train", "student_test"):
        split_directory = root / dataset_split
        if not split_directory.is_dir():
            issues.append(
                _issue("", dataset_split, "missing_split", str(split_directory))
            )
            continue
        for trip_directory in _trip_directories(split_directory):
            trip_id = trip_directory.name
            track_path = trip_directory / f"{trip_id}.track"
            nodes_path = trip_directory / f"{trip_id}.nodes"
            arcs_path = trip_directory / f"{trip_id}.arcs"
            route_path = trip_directory / f"{trip_id}.route"
            try:
                track = load_track(track_path)
            except (OSError, ValueError) as error:
                issues.append(
                    _issue(trip_id, dataset_split, "invalid_track", str(error))
                )
                continue

            if not nodes_path.is_file():
                issues.append(
                    _issue(
                        trip_id,
                        dataset_split,
                        "missing_nodes",
                        f"Required node file was not found: {nodes_path}",
                    )
                )
                observations = _empty_observation_rows(
                    track, trip_id, dataset_split, "missing_nodes"
                )
                if dataset_split == "train":
                    train_observations.append(observations)
                else:
                    student_observations.append(observations)
                continue
            try:
                nodes = load_nodes(nodes_path)
            except (OSError, ValueError) as error:
                issues.append(_issue(trip_id, dataset_split, "invalid_nodes", str(error)))
                continue

            if not arcs_path.is_file():
                issues.append(
                    _issue(
                        trip_id,
                        dataset_split,
                        "missing_arcs",
                        f"Required arc file was not found: {arcs_path}",
                    )
                )
                observations = _empty_observation_rows(
                    track, trip_id, dataset_split, "missing_arcs"
                )
                if dataset_split == "train":
                    train_observations.append(observations)
                else:
                    student_observations.append(observations)
                continue
            try:
                arcs = load_arcs(arcs_path)
            except (OSError, ValueError) as error:
                issues.append(_issue(trip_id, dataset_split, "invalid_arcs", str(error)))
                continue

            route: np.ndarray | None = None
            labels: pd.DataFrame | None = None
            if dataset_split == "train":
                if not route_path.is_file():
                    issues.append(
                        _issue(
                            trip_id,
                            dataset_split,
                            "missing_route",
                            f"Required training route file was not found: {route_path}",
                        )
                    )
                    observations = _empty_observation_rows(
                        track, trip_id, dataset_split, "missing_route"
                    )
                    train_observations.append(observations)
                    continue
                try:
                    route = load_route(route_path)
                    reference_candidates = build_route_reference_candidates(
                        track,
                        nodes,
                        arcs,
                        route,
                        trip_id,
                        reference_radius_m=search_radius_m,
                    )
                    labels = align_route_labels(
                        route,
                        arcs,
                        reference_candidates,
                        len(track),
                        trip_id,
                        ambiguity_margin_m=ambiguity_margin_m,
                    )
                except (OSError, ValueError) as error:
                    issues.append(
                        _issue(trip_id, dataset_split, "invalid_route_alignment", str(error))
                    )
                    observations = _empty_observation_rows(
                        track, trip_id, dataset_split, "invalid_route_alignment"
                    )
                    train_observations.append(observations)
                    continue

                components = split_route_components(route, arcs)
                route_summaries.append(
                    {
                        "trip_id": trip_id,
                        "route_position_count": int(len(route)),
                        "route_component_count": int(len(components)),
                        "disconnected_transition_count": int(max(0, len(components) - 1)),
                        "repeated_route_position_count": int(
                            len(route) - len(np.unique(route))
                        ),
                        "labelled_observation_count": int(
                            (labels["label_status"] == "labelled").sum()
                        ),
                        "ambiguous_observation_count": int(
                            (labels["label_status"] == "ambiguous").sum()
                        ),
                        "unlabelled_observation_count": int(
                            (labels["label_status"] != "labelled").sum()
                        ),
                    }
                )
            try:
                candidate_results_by_radius = {
                    radius: generate_candidates(
                        track,
                        nodes,
                        arcs,
                        trip_id,
                        search_radius_m=radius,
                        max_candidates=max_candidates,
                    )
                    for radius in evaluation_radii_m
                }
            except (OSError, ValueError) as error:
                issues.append(
                    _issue(trip_id, dataset_split, "candidate_generation_error", str(error))
                )
                observations = _empty_observation_rows(
                    track, trip_id, dataset_split, "candidate_generation_error"
                )
                if dataset_split == "train":
                    train_observations.append(observations)
                else:
                    student_observations.append(observations)
                continue

            default_candidates = candidate_results_by_radius[
                float(search_radius_m)
            ].candidates
            try:
                features = extract_candidate_features(
                    track, nodes, arcs, default_candidates, trip_id
                )
            except (OSError, ValueError) as error:
                issues.append(
                    _issue(trip_id, dataset_split, "feature_extraction_error", str(error))
                )
                continue
            features["dataset_split"] = dataset_split
            if labels is not None:
                features = features.merge(
                    labels,
                    on=["trip_id", "gps_observation_index"],
                    how="left",
                    validate="many_to_one",
                )
                features["is_match"] = pd.array(
                    [
                        None
                        if status != "labelled"
                        else int(arc_id == label_arc_id)
                        for arc_id, label_arc_id, status in zip(
                            features["candidate_arc_id"],
                            features["label_arc_id"],
                            features["label_status"],
                        )
                    ],
                    dtype="Int8",
                )
                train_features.append(features)
                observations = candidate_results_by_radius[
                    float(search_radius_m)
                ].observations.merge(
                    labels,
                    on=["trip_id", "gps_observation_index"],
                    how="left",
                    validate="one_to_one",
                )
                observations["dataset_split"] = dataset_split
                observations["processing_status"] = "processed"
                train_observations.append(observations)

                candidate_sets = {
                    radius: result.candidates
                    for radius, result in candidate_results_by_radius.items()
                }
                summary = candidate_recall_summary(
                    labels, candidate_sets, len(track)
                )
                summary["trip_id"] = trip_id
                summary["gps_observation_count"] = len(track)
                trip_evaluation_summaries.append(summary)
                for invalid_arc_id in candidate_results_by_radius[
                    float(search_radius_m)
                ].invalid_arc_ids:
                    issues.append(
                        _issue(
                            trip_id,
                            dataset_split,
                            "invalid_arc_geometry",
                            f"Arc row {invalid_arc_id} has invalid endpoint references or coordinates",
                        )
                    )
            else:
                student_features.append(features)
                observations = candidate_results_by_radius[
                    float(search_radius_m)
                ].observations.copy()
                observations["dataset_split"] = dataset_split
                observations["processing_status"] = "processed"
                student_observations.append(observations)
                for invalid_arc_id in candidate_results_by_radius[
                    float(search_radius_m)
                ].invalid_arc_ids:
                    issues.append(
                        _issue(
                            trip_id,
                            dataset_split,
                            "invalid_arc_geometry",
                            f"Arc row {invalid_arc_id} has invalid endpoint references or coordinates",
                        )
                    )

    train_frame = (
        pd.concat(train_features, ignore_index=True)
        if train_features
        else pd.DataFrame(columns=feature_columns)
    )
    student_frame = (
        pd.concat(student_features, ignore_index=True)
        if student_features
        else pd.DataFrame(columns=feature_columns)
    )
    train_observation_frame = (
        pd.concat(train_observations, ignore_index=True)
        if train_observations
        else pd.DataFrame()
    )
    student_observation_frame = (
        pd.concat(student_observations, ignore_index=True)
        if student_observations
        else pd.DataFrame()
    )
    recall_summary = _weighted_recall_summary(
        trip_evaluation_summaries, evaluation_radii_m
    )
    route_summary_frame = pd.DataFrame(route_summaries)
    issue_frame = pd.DataFrame(
        issues, columns=("trip_id", "dataset_split", "issue_type", "description")
    )

    artifacts = {
        "train_candidate_features": train_frame,
        "student_test_candidate_features": student_frame,
        "train_observations": train_observation_frame,
        "student_test_observations": student_observation_frame,
        "candidate_recall": recall_summary,
        "route_summary": route_summary_frame,
        "processing_issues": issue_frame,
    }
    for artifact_name, frame in artifacts.items():
        frame.to_csv(output / f"{artifact_name}.csv", index=False)
    return artifacts
