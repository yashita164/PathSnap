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
    assert result.predictions["decode_segment"].nunique() == 1
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
    previous_arc = int(result.predictions.loc[0, "hmm_arc_id"])
    next_arc = int(result.predictions.loc[1, "hmm_arc_id"])
    assert arcs.iloc[previous_arc]["to_node"] != arcs.iloc[next_arc]["from_node"]
    assert result.route_continuity == 0.0


def test_finite_transition_from_pruned_state_does_not_revive_path() -> None:
    nodes = pd.DataFrame(
        {
            "longitude": [0.0, 0.001, 0.002, 0.01, 0.011, 0.012],
            "latitude": [0.0] * 6,
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node": [0, 1, 3, 4],
            "to_node": [1, 2, 4, 5],
        },
        dtype=np.int64,
    )
    track = pd.DataFrame(
        {
            "longitude": [0.0005, 0.0015, 0.0015],
            "latitude": [0.0, 0.0, 0.0],
            "timestamp": [1.0, 2.0, 3.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0, 1, 1, 2],
            "candidate_arc_id": [0, 1, 2, 3],
            "candidate_probability": [0.9, 0.9, 0.01, 0.9],
        }
    )

    result = decode_route(track, nodes, arcs, candidates)

    predictions = result.predictions
    assert predictions["hmm_arc_id"].tolist() == [0, 1, 3]
    assert predictions["decode_segment"].tolist() == [0, 0, 1]
    assert arcs.iloc[1]["to_node"] != arcs.iloc[3]["from_node"]
    assert result.route_continuity == 0.5


def test_invalid_arc_endpoint_is_not_selected_over_valid_candidate() -> None:
    nodes, _ = _network()
    arcs = pd.DataFrame(
        {
            "from_node": [0, 1, 0],
            "to_node": [1, 2, len(nodes)],
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
            "gps_observation_index": [0, 0, 1],
            "candidate_arc_id": [0, 2, 1],
            "candidate_probability": [0.4, 0.99, 0.8],
        }
    )

    result = decode_route(track, nodes, arcs, candidates)

    assert result.predictions["classifier_only_arc_id"].tolist() == [0, 1]
    assert result.predictions["hmm_arc_id"].tolist() == [0, 1]
    assert result.predictions["decode_status"].tolist() == ["decoded", "decoded"]


def test_only_invalid_arc_candidates_get_no_valid_candidates_status() -> None:
    nodes, _ = _network()
    arcs = pd.DataFrame({"from_node": [0], "to_node": [len(nodes)]})
    track = pd.DataFrame(
        {
            "longitude": [0.0005],
            "latitude": [0.0],
            "timestamp": [1.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0],
            "candidate_arc_id": [0],
            "candidate_probability": [0.9],
        }
    )

    result = decode_route(track, nodes, arcs, candidates)

    assert result.predictions["classifier_only_arc_id"].tolist() == [-1]
    assert result.predictions["hmm_arc_id"].tolist() == [-1]
    assert result.predictions["decode_status"].tolist() == ["no_valid_candidates"]
    assert np.isnan(result.route_continuity)


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
