"""Focused tests for Phase 3 candidate generation, features, and labels."""

import numpy as np
import pandas as pd

from src.candidates import generate_candidates, project_trip_coordinates
from src.features import extract_candidate_features
from src.geo import point_to_segment_distance
from src.labeling import align_route_labels, split_route_components


def _network() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "longitude": [-0.001, 0.0, 0.001, 0.001, 0.002],
            "latitude": [0.0, 0.0, 0.0, 0.001, 0.001],
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node": [0, 1, 2, 3],
            "to_node": [1, 2, 3, 4],
        },
        dtype=np.int64,
    )
    return nodes, arcs


def _track() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [-0.0005, 0.0005, 0.0015],
            "latitude": [0.00005, 0.00005, 0.00105],
            "timestamp": [1.0, 2.0, 3.0],
        }
    )


def test_candidate_index_matches_brute_force_and_preserves_arc_ids() -> None:
    nodes, arcs = _network()
    track = _track()
    projected = project_trip_coordinates(track, nodes)

    result = generate_candidates(
        track, nodes, arcs, "trip", search_radius_m=30.0, max_candidates=8
    )
    first_observation = result.candidates.loc[
        result.candidates["gps_observation_index"] == 0
    ]
    point = projected.gps_xy[0]
    endpoints = arcs.iloc[0][["from_node", "to_node"]].to_numpy(dtype=np.int64)
    expected_distance = point_to_segment_distance(
        point,
        projected.node_xy[endpoints[0]],
        projected.node_xy[endpoints[1]],
    )
    assert 0 in first_observation["candidate_arc_id"].to_numpy()
    assert np.isclose(
        first_observation.loc[
            first_observation["candidate_arc_id"] == 0, "distance_m"
        ].iloc[0],
        expected_distance,
    )
    assert result.candidates.groupby("gps_observation_index").size().max() <= 8
    assert result.invalid_arc_ids == ()


def test_features_use_metric_units_and_flag_unavailable_heading() -> None:
    nodes, arcs = _network()
    track = _track()
    candidates = generate_candidates(track, nodes, arcs, "trip", 30.0, 8).candidates

    features = extract_candidate_features(track, nodes, arcs, candidates, "trip")

    assert np.isfinite(features.select_dtypes(include="number").to_numpy()).all()
    assert features["candidate_arc_length_m"].max() > 100.0
    first_observation = features.loc[features["gps_observation_index"] == 0]
    assert (first_observation["gps_heading_available"] == 0).all()
    moving_observation = features.loc[features["gps_observation_index"] == 1]
    assert (moving_observation["gps_heading_available"] == 1).all()
    assert moving_observation["gps_speed_mps"].iloc[0] > 0


def test_invalid_time_intervals_do_not_create_speed_features() -> None:
    nodes, arcs = _network()
    track = _track()
    track["timestamp"] = [1.0, 1.0, 0.0]
    candidates = generate_candidates(track, nodes, arcs, "trip", 30.0, 8).candidates

    features = extract_candidate_features(track, nodes, arcs, candidates, "trip")

    duplicate_time = features.loc[features["gps_observation_index"] == 1]
    negative_time = features.loc[features["gps_observation_index"] == 2]
    assert (duplicate_time["time_interval_valid"] == 0).all()
    assert (duplicate_time["gps_speed_available"] == 0).all()
    assert (negative_time["time_interval_valid"] == 0).all()
    assert (negative_time["gps_speed_available"] == 0).all()
    assert np.isfinite(features["gps_speed_mps"]).all()


def test_alignment_is_monotonic_and_allows_repeated_route_arcs() -> None:
    arcs = pd.DataFrame(
        {"from_node": [0, 1], "to_node": [0, 2]}, dtype=np.int64
    )
    route = np.array([0, 0, 1], dtype=np.int64)
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0, 1, 2, 3],
            "candidate_arc_id": [0, 0, 1, 0],
            "distance_m": [1.0, 2.0, 0.1, 20.0],
        }
    )

    labels = align_route_labels(route, arcs, candidates, 4, "trip")

    assert labels["label_status"].tolist() == [
        "labelled",
        "labelled",
        "labelled",
        "infeasible",
    ]
    labelled_positions = labels.loc[
        labels["label_status"] == "labelled", "label_route_position"
    ].astype(int)
    assert labelled_positions.is_monotonic_increasing


def test_disconnected_components_and_close_competitors_remain_unlabelled() -> None:
    arcs = pd.DataFrame(
        {"from_node": [0, 1, 8], "to_node": [1, 2, 9]}, dtype=np.int64
    )
    route = np.array([0, 1, 2], dtype=np.int64)
    candidates = pd.DataFrame(
        {
            "gps_observation_index": [0, 0, 1, 2],
            "candidate_arc_id": [0, 1, 1, 2],
            "distance_m": [10.0, 12.0, 10.0, 10.0],
        }
    )

    components = split_route_components(route, arcs)
    labels = align_route_labels(route, arcs, candidates, 3, "trip")

    assert [component.tolist() for component in components] == [[0, 1], [2]]
    assert labels.loc[0, "label_status"] == "ambiguous"
    assert labels.loc[1, "label_status"] == "labelled"
    assert labels.loc[2, "label_status"] == "labelled"
    assert labels.loc[1, "alignment_component"] == 0
    assert labels.loc[2, "alignment_component"] == 1
