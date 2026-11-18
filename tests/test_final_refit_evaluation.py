import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.model_evaluation import prepare_evaluation_data
from src.optimization.finalize_logistic_regression import score_predictions


def test_score_predictions_reports_candidate_misses_and_observation_top1() -> None:
    metadata = pd.DataFrame(
        {
            "trip_id": ["trip-a", "trip-a", "trip-a"],
            "gps_observation_index": [0, 0, 1],
            "candidate_arc_id": [3, 5, 7],
            "label_arc_id": [5, 5, 9],
        }
    )
    labels = pd.DataFrame(
        {
            "trip_id": ["trip-a", "trip-a"],
            "gps_observation_index": [0, 1],
            "label_arc_id": [5, 9],
        }
    )
    result = score_predictions(
        metadata,
        labels,
        predictions=np.array([0, 1, 0]),
        probabilities=np.array([0.2, 0.8, 0.1]),
        targets=np.array([0, 1, 0]),
    )

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["top1_accuracy"] == pytest.approx(1.0)
    assert result["top1_observations_evaluated"] == 1
    assert result["labelled_observations"] == 2
    assert result["candidate_miss_count"] == 1
    assert result["candidate_miss_rate"] == pytest.approx(0.5)


def test_score_predictions_rejects_misaligned_probability_rows() -> None:
    metadata = pd.DataFrame(
        {
            "trip_id": ["trip-a"],
            "gps_observation_index": [0],
            "candidate_arc_id": [3],
            "label_arc_id": [3],
        }
    )
    labels = pd.DataFrame(
        {
            "trip_id": ["trip-a"],
            "gps_observation_index": [0],
            "label_arc_id": [3],
        }
    )

    with pytest.raises(ValueError, match="must align"):
        score_predictions(
            metadata,
            labels,
            predictions=np.array([1]),
            probabilities=np.array([], dtype=float),
            targets=np.array([1]),
        )


def test_training_preparation_rejects_student_test_rows() -> None:
    candidates = pd.DataFrame(
        [
            {
                **{column: 0.0 for column in FEATURE_COLUMNS},
                "trip_id": "student-trip",
                "gps_observation_index": 0,
                "candidate_arc_id": arc_id,
                "dataset_split": "student_test",
                "label_arc_id": 0,
                "label_status": "labelled",
                "is_match": int(arc_id == 0),
            }
            for arc_id in (0, 1)
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "trip_id": "student-trip",
                "gps_observation_index": 0,
                "has_valid_coordinates": True,
                "label_arc_id": 0,
                "label_status": "labelled",
                "dataset_split": "student_test",
                "processing_status": "processed",
            }
        ]
    )

    with pytest.raises(ValueError, match="only the training split"):
        prepare_evaluation_data(candidates, observations)
