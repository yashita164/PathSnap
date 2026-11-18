import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.model_evaluation import prepare_evaluation_data
from src.optimization.optuna_search import (
    INNER_SPLITS,
    OUTER_SPLITS,
    _group_kfold_splits,
    _score_c_on_folds,
    optimize_logistic_regression,
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


def test_nested_group_splits_never_share_trips() -> None:
    candidates, observations = _synthetic_training_data()
    prepared = prepare_evaluation_data(candidates, observations)
    outer_splits = _group_kfold_splits(
        prepared.X,
        prepared.y,
        prepared.metadata["trip_id"],
        OUTER_SPLITS,
    )

    assert len(outer_splits) == OUTER_SPLITS
    for outer_train, outer_valid in outer_splits:
        outer_groups = set(prepared.metadata.iloc[outer_train]["trip_id"])
        outer_validation_groups = set(prepared.metadata.iloc[outer_valid]["trip_id"])
        assert outer_groups.isdisjoint(outer_validation_groups)

        X_outer_train = prepared.X.iloc[outer_train].reset_index(drop=True)
        y_outer_train = prepared.y[outer_train]
        groups_outer_train = prepared.metadata.iloc[outer_train]["trip_id"].reset_index(
            drop=True
        )
        inner_splits = _group_kfold_splits(
            X_outer_train,
            y_outer_train,
            groups_outer_train,
            INNER_SPLITS,
        )
        assert len(inner_splits) == INNER_SPLITS
        for inner_train, inner_valid in inner_splits:
            train_groups = set(groups_outer_train.iloc[inner_train])
            validation_groups = set(groups_outer_train.iloc[inner_valid])
            assert train_groups.isdisjoint(validation_groups)


def test_top1_objective_scores_all_inner_folds() -> None:
    candidates, observations = _synthetic_training_data()
    prepared = prepare_evaluation_data(candidates, observations)
    outer_train, _ = _group_kfold_splits(
        prepared.X,
        prepared.y,
        prepared.metadata["trip_id"],
        OUTER_SPLITS,
    )[0]
    X_outer_train = prepared.X.iloc[outer_train].reset_index(drop=True)
    y_outer_train = prepared.y[outer_train]
    metadata_outer_train = prepared.metadata.iloc[outer_train].reset_index(drop=True)
    inner_splits = _group_kfold_splits(
        X_outer_train,
        y_outer_train,
        metadata_outer_train["trip_id"],
        INNER_SPLITS,
    )

    objective, top1_scores, log_losses = _score_c_on_folds(
        1.0,
        X_outer_train,
        y_outer_train,
        metadata_outer_train,
        prepared.labelled_observations,
        inner_splits,
    )

    assert len(top1_scores) == INNER_SPLITS
    assert len(log_losses) == INNER_SPLITS
    assert 0.0 <= objective <= 1.0
    assert np.isclose(objective, np.mean(top1_scores))


def test_seeded_search_is_reproducible_and_records_trials(tmp_path) -> None:
    candidates, observations = _synthetic_training_data()
    first = optimize_logistic_regression(
        candidates,
        observations,
        tmp_path / "first",
        tmp_path / "first_figures",
        n_trials=5,
        seed=17,
    )
    second = optimize_logistic_regression(
        candidates,
        observations,
        tmp_path / "second",
        tmp_path / "second_figures",
        n_trials=5,
        seed=17,
    )

    first_trials = first["trials"].sort_values(["outer_fold", "trial_number"])
    second_trials = second["trials"].sort_values(["outer_fold", "trial_number"])
    pd.testing.assert_frame_equal(
        first_trials[
            ["outer_fold", "trial_number", "C", "objective_value", "status"]
        ].reset_index(drop=True),
        second_trials[
            ["outer_fold", "trial_number", "C", "objective_value", "status"]
        ].reset_index(drop=True),
    )
    assert len(first_trials) == 5
    assert first_trials["status"].eq("COMPLETE").all()
    assert first["best_C"] == second["best_C"]
    assert (tmp_path / "first" / "optuna_trials.csv").is_file()
    assert (tmp_path / "first" / "best_parameters.json").is_file()
    assert (tmp_path / "first" / "optimization_summary.md").is_file()
    assert (
        tmp_path / "first_figures" / "phase6_optimization_history.png"
    ).is_file()
