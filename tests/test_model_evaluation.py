import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.model_evaluation import (
    _model_specs,
    _classification_metrics,
    _top1_metrics,
    evaluate_models,
    prepare_evaluation_data,
)


def _synthetic_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    candidate_rows = []
    observation_rows = []
    for trip_number in range(10):
        trip_id = f"trip-{trip_number:02d}"
        for observation_index in range(4):
            correct_arc_id = observation_index * 3
            observation_rows.append(
                {
                    "trip_id": trip_id,
                    "gps_observation_index": observation_index,
                    "has_valid_coordinates": True,
                    "label_arc_id": correct_arc_id,
                    "label_status": "labelled",
                    "dataset_split": "train",
                    "processing_status": "processed",
                }
            )
            for candidate_offset in range(3):
                candidate_arc_id = observation_index * 3 + candidate_offset
                is_match = int(candidate_arc_id == correct_arc_id)
                values = rng.normal(size=len(FEATURE_COLUMNS))
                values[0] += 2.0 * is_match
                row = dict(zip(FEATURE_COLUMNS, values))
                row.update(
                    {
                        "trip_id": trip_id,
                        "gps_observation_index": observation_index,
                        "candidate_arc_id": candidate_arc_id,
                        "dataset_split": "train",
                        "label_arc_id": correct_arc_id,
                        "label_status": "labelled",
                        "is_match": is_match,
                    }
                )
                candidate_rows.append(row)
    return pd.DataFrame(candidate_rows), pd.DataFrame(observation_rows)


def test_model_registry_contains_the_five_existing_classifiers() -> None:
    assert [spec.name for spec in _model_specs()] == [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "MLP",
        "Distance-weighted KNN",
    ]


def test_classification_metrics_match_known_results() -> None:
    metrics = _classification_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0.1, 0.8, 0.6, 0.2]),
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert np.isclose(metrics["f1"], 2 / 3)
    assert metrics["roc_auc"] == 1.0
    assert np.isfinite(metrics["log_loss"])


def test_top1_ties_and_multiple_positive_rows_are_explicit() -> None:
    metadata = pd.DataFrame(
        {
            "trip_id": ["trip-a"] * 5,
            "gps_observation_index": [0, 0, 0, 1, 1],
            "candidate_arc_id": [2, 2, 1, 5, 6],
            "label_arc_id": [2, 2, 2, 5, 5],
        }
    )
    observations = pd.DataFrame(
        {
            "trip_id": ["trip-a", "trip-a"],
            "gps_observation_index": [0, 1],
            "label_arc_id": [2, 5],
        }
    )

    accuracy, evaluated, unavailable, multiple, ties = _top1_metrics(
        metadata,
        observations,
        np.array([0.5, 0.5, 0.5, 0.9, 0.1]),
        {"trip-a"},
        np.array([1, 1, 0, 1, 0]),
    )

    assert accuracy == 0.5
    assert evaluated == 2
    assert unavailable == 0
    assert multiple == 1
    assert ties == 1


def test_preparation_excludes_metadata_and_unreliable_labels() -> None:
    candidates, observations = _synthetic_training_data()
    unreliable = candidates.iloc[[0]].copy()
    unreliable["label_status"] = "ambiguous"
    unreliable["is_match"] = np.nan
    candidates = pd.concat([candidates, unreliable], ignore_index=True)

    prepared = prepare_evaluation_data(candidates, observations)

    assert tuple(prepared.X.columns) == tuple(FEATURE_COLUMNS)
    assert not {"trip_id", "candidate_arc_id", "label_arc_id", "label_status", "is_match"}.intersection(prepared.X.columns)
    assert len(prepared.X) == len(candidates) - 1
    assert prepared.exclusions["candidate_rows_excluded_by_label_status"]["ambiguous"] == 1


def test_five_models_share_grouped_splits_and_generate_outputs(tmp_path) -> None:
    candidates, observations = _synthetic_training_data()

    result = evaluate_models(
        candidates,
        observations,
        tmp_path,
        n_splits=5,
        inference_batch_size=16,
    )

    assert len(result.fold_results) == 25
    assert len(result.comparison) == 5
    assert set(result.fold_results["trip_overlap_count"]) == {0}
    for _, fold in result.fold_results.groupby("fold"):
        assert fold["validation_trip_ids"].nunique() == 1
    assert (result.fold_results["n_validation_samples"] > 0).all()
    assert (result.fold_results["top1_observations_evaluated"] > 0).all()
    assert result.fold_results[
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "top1_accuracy",
        ]
    ].notna().all().all()
    assert (tmp_path / "model_fold_results.csv").is_file()
    assert (tmp_path / "model_comparison.csv").is_file()
    assert (tmp_path / "model_evaluation_summary.md").is_file()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5
