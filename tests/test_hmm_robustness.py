import numpy as np
import pandas as pd
import pytest

from src.hmm import decode_route


def _linear_network(arc_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "longitude": np.arange(arc_count + 1, dtype=np.float64) * 0.0001,
            "latitude": np.zeros(arc_count + 1, dtype=np.float64),
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node": np.arange(arc_count, dtype=np.int64),
            "to_node": np.arange(1, arc_count + 1, dtype=np.int64),
        }
    )
    return nodes, arcs


@pytest.mark.parametrize("observation_count", [1, 4, 9])
def test_varying_observation_and_candidate_counts(
    observation_count: int,
) -> None:
    nodes, arcs = _linear_network(6)
    track = pd.DataFrame(
        {
            "longitude": np.full(observation_count, 0.00005),
            "latitude": np.zeros(observation_count),
            "timestamp": np.arange(observation_count, dtype=np.float64) + 1.0,
        }
    )
    candidate_counts = [1, 4, 2, 6, 3, 5, 1, 2, 4][:observation_count]
    observation_indices = np.concatenate(
        [
            np.full(count, index, dtype=np.int64)
            for index, count in enumerate(candidate_counts)
        ]
    )
    arc_ids = np.concatenate(
        [np.arange(count, dtype=np.int64) for count in candidate_counts]
    )
    probabilities = np.random.default_rng(20261118).uniform(
        0.05, 0.95, size=len(arc_ids)
    )
    candidates = pd.DataFrame(
        {
            "gps_observation_index": observation_indices,
            "candidate_arc_id": arc_ids,
        }
    )

    result = decode_route(track, nodes, arcs, candidates, probabilities)

    assert len(result.predictions) == observation_count
    assert result.predictions["hmm_arc_id"].between(0, len(arcs) - 1).all()
    assert result.predictions["decode_status"].eq("decoded").all()
    for observation_index, count in enumerate(candidate_counts):
        assert int(
            result.predictions.loc[observation_index, "classifier_only_arc_id"]
        ) in range(count)


def test_single_observation_decodes_highest_probability_candidate() -> None:
    nodes, arcs = _linear_network(3)
    track = pd.DataFrame(
        {"longitude": [0.00015], "latitude": [0.0], "timestamp": [5.0]}
    )
    candidates = pd.DataFrame(
        {"gps_observation_index": [0, 0], "candidate_arc_id": [0, 1]}
    )

    result = decode_route(track, nodes, arcs, candidates, [0.1, 0.9])

    assert result.predictions["hmm_arc_id"].tolist() == [1]
    assert result.predictions["decode_segment"].tolist() == [0]
    assert np.isnan(result.route_continuity)


def test_very_small_probabilities_remain_decodable() -> None:
    nodes, arcs = _linear_network(3)
    track = pd.DataFrame(
        {"longitude": [0.00015], "latitude": [0.0], "timestamp": [5.0]}
    )
    candidates = pd.DataFrame(
        {"gps_observation_index": [0, 0], "candidate_arc_id": [1, 0]}
    )

    result = decode_route(track, nodes, arcs, candidates, [1e-300, 2e-300])

    assert result.predictions["hmm_arc_id"].tolist() == [0]
    assert np.isfinite(result.predictions["hmm_candidate_probability"]).all()
    assert result.predictions["hmm_candidate_probability"].iloc[0] == 2e-300


def test_empty_candidate_sequence_preserves_all_observation_rows() -> None:
    nodes, arcs = _linear_network(3)
    track = pd.DataFrame(
        {
            "longitude": [0.00005, 0.00015, 0.00025],
            "latitude": [0.0, 0.0, 0.0],
            "timestamp": [1.0, 2.0, 3.0],
        }
    )
    candidates = pd.DataFrame(
        columns=["gps_observation_index", "candidate_arc_id"]
    )

    result = decode_route(track, nodes, arcs, candidates)

    assert result.predictions["gps_observation_index"].tolist() == [0, 1, 2]
    assert result.predictions["hmm_arc_id"].tolist() == [-1, -1, -1]
    assert result.predictions["decode_status"].tolist() == [
        "no_candidates",
        "no_candidates",
        "no_candidates",
    ]
    assert np.isnan(result.route_continuity)


def test_equal_probability_paths_are_stable_and_continuous() -> None:
    nodes = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.001, 0.002],
            "latitude": [0.0, 0.0, 0.0, 0.0],
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node": [0, 0, 1, 2],
            "to_node": [1, 2, 3, 3],
        },
        dtype=np.int64,
    )
    track = pd.DataFrame(
        {
            "longitude": [0.0005, 0.0015],
            "latitude": [0.0, 0.0],
            "timestamp": [1.0, 2.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0, 0, 1, 1],
            "candidate_arc_id": [0, 1, 2, 3],
        }
    )

    first = decode_route(track, nodes, arcs, candidates, [0.5] * 4)
    second = decode_route(track, nodes, arcs, candidates, [0.5] * 4)

    assert first.predictions["hmm_arc_id"].tolist() == [0, 2]
    assert second.predictions["hmm_arc_id"].tolist() == [0, 2]
    assert first.route_continuity == 1.0
