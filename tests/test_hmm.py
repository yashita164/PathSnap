import numpy as np
import pandas as pd

from src.hmm import calculate_route_continuity, decode_route


def _network() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.002, 0.01, 0.011, 0.012, 0.013],
            "latitude": [0.0] * 7,
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node": [0, 1, 3, 5],
            "to_node": [1, 2, 4, 6],
        },
        dtype=np.int64,
    )
    return nodes, arcs


def test_viterbi_prefers_connected_route_over_independent_top_scores() -> None:
    nodes, arcs = _network()
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
            "candidate_arc_id": [0, 2, 1, 3],
        }
    )
    classifier_probabilities = np.array([0.45, 0.8, 0.45, 0.8])

    result = decode_route(
        track, nodes, arcs, candidates, classifier_probabilities
    )

    assert result.predictions["classifier_only_arc_id"].tolist() == [2, 3]
    assert result.predictions["hmm_arc_id"].tolist() == [0, 1]
    assert result.route_continuity == 1.0


def test_empty_candidates_and_large_time_gaps_split_decode_runs() -> None:
    nodes, arcs = _network()
    track = pd.DataFrame(
        {
            "longitude": [0.0002, 0.0005, 0.0008],
            "latitude": [0.0, 0.0, 0.0],
            "timestamp": [1.0, 2.0, 100.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0, 2],
            "candidate_arc_id": [0, 0],
            "candidate_probability": [0.8, 0.7],
        }
    )

    result = decode_route(track, nodes, arcs, candidates)
    predictions = result.predictions

    assert predictions["hmm_arc_id"].tolist() == [0, -1, 0]
    assert predictions["decode_status"].tolist() == [
        "decoded",
        "no_candidates",
        "decoded",
    ]
    assert predictions.loc[0, "decode_segment"] != predictions.loc[2, "decode_segment"]
    assert np.isnan(result.route_continuity)


def test_unreachable_transition_decodes_each_observation_independently() -> None:
    nodes, arcs = _network()
    track = pd.DataFrame(
        {
            "longitude": [0.0005, 0.0015],
            "latitude": [0.0, 0.0],
            "timestamp": [1.0, 2.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0, 1],
            "candidate_arc_id": [2, 3],
            "candidate_probability": [0.6, 0.7],
        }
    )

    result = decode_route(track, nodes, arcs, candidates)

    assert result.predictions["hmm_arc_id"].tolist() == [2, 3]
    assert result.predictions["decode_segment"].nunique() == 2
    assert result.route_continuity == 0.0


def test_large_gap_is_excluded_from_route_continuity() -> None:
    nodes, arcs = _network()
    track = pd.DataFrame(
        {
            "longitude": [0.0005, 0.0015],
            "latitude": [0.0, 0.0],
            "timestamp": [1.0, 100.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0, 1],
            "candidate_arc_id": [2, 3],
            "candidate_probability": [0.6, 0.7],
        }
    )

    result = decode_route(track, nodes, arcs, candidates)

    assert result.predictions["hmm_arc_id"].tolist() == [2, 3]
    assert result.predictions["decode_segment"].nunique() == 2
    assert np.isnan(result.route_continuity)


def test_empty_candidate_table_produces_missing_predictions() -> None:
    nodes, arcs = _network()
    track = pd.DataFrame(
        {
            "longitude": [0.0005],
            "latitude": [0.0],
            "timestamp": [1.0],
        }
    )
    candidates = pd.DataFrame(
        columns=["gps_observation_index", "candidate_arc_id"]
    )

    result = decode_route(track, nodes, arcs, candidates)

    assert result.predictions["hmm_arc_id"].tolist() == [-1]
    assert result.predictions["decode_status"].tolist() == ["no_candidates"]
    assert np.isnan(result.route_continuity)


def test_route_continuity_uses_only_valid_consecutive_pairs() -> None:
    _, arcs = _network()

    assert calculate_route_continuity([0, 1], arcs) == 1.0
    assert calculate_route_continuity([0, 2], arcs) == 0.0
    assert np.isnan(calculate_route_continuity([0, 1], arcs, [0, 1]))
