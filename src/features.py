"""Label-independent feature extraction for GPS/candidate-arc pairs."""

import numpy as np
import pandas as pd

from src.geo import angular_difference, calculate_bearing, calculate_distance


IDENTIFIER_COLUMNS = (
    "trip_id",
    "gps_observation_index",
    "candidate_arc_id",
)
FEATURE_COLUMNS = (
    "point_to_segment_distance_m",
    "distance_to_start_node_m",
    "distance_to_end_node_m",
    "candidate_arc_length_m",
    "candidate_road_bearing_deg",
    "candidate_bearing_available",
    "gps_movement_bearing_deg",
    "gps_heading_available",
    "bearing_difference_deg",
    "heading_consistency",
    "gps_speed_mps",
    "gps_speed_kmh",
    "gps_speed_available",
    "time_delta_s",
    "time_interval_valid",
    "time_gap_over_5s",
    "time_gap_over_60s",
)

_METRIC_FEATURES = (
    "point_to_segment_distance_m",
    "distance_to_start_node_m",
    "distance_to_end_node_m",
    "candidate_arc_length_m",
    "gps_speed_mps",
    "gps_speed_kmh",
    "time_delta_s",
)


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _integer_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"Candidate column '{column}' must contain integers")
    return values.astype(np.int64)


def _valid_coordinates(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )


def _gps_movement_features(
    track: pd.DataFrame,
    minimum_heading_speed_mps: float,
) -> dict[str, np.ndarray]:
    longitude = _numeric_values(track, "longitude")
    latitude = _numeric_values(track, "latitude")
    timestamps = _numeric_values(track, "timestamp")
    valid_coordinates = _valid_coordinates(longitude, latitude)
    observation_count = len(track)

    time_delta = np.zeros(observation_count, dtype=np.float64)
    time_interval_valid = np.zeros(observation_count, dtype=bool)
    distance_m = np.zeros(observation_count, dtype=np.float64)
    gps_speed_mps = np.zeros(observation_count, dtype=np.float64)
    gps_speed_available = np.zeros(observation_count, dtype=bool)
    gps_heading_deg = np.zeros(observation_count, dtype=np.float64)
    gps_heading_available = np.zeros(observation_count, dtype=bool)
    if observation_count > 1:
        elapsed = timestamps[1:] - timestamps[:-1]
        valid_elapsed = (
            np.isfinite(timestamps[1:])
            & np.isfinite(timestamps[:-1])
            & np.isfinite(elapsed)
            & (elapsed > 0)
        )
        time_delta[1:][valid_elapsed] = elapsed[valid_elapsed]
        time_interval_valid[1:] = valid_elapsed

        valid_coordinate_pairs = valid_coordinates[1:] & valid_coordinates[:-1]
        valid_pairs = valid_elapsed & valid_coordinate_pairs
        valid_indices = np.flatnonzero(valid_pairs)
        if valid_indices.size:
            previous_indices = valid_indices
            current_indices = valid_indices + 1
            distances = np.asarray(
                calculate_distance(
                    longitude[previous_indices],
                    latitude[previous_indices],
                    longitude[current_indices],
                    latitude[current_indices],
                ),
                dtype=np.float64,
            )
            speeds = distances / elapsed[valid_indices]
            distance_m[current_indices] = distances
            gps_speed_mps[current_indices] = speeds
            gps_speed_available[current_indices] = np.isfinite(speeds)
            moving = np.isfinite(speeds) & (speeds >= minimum_heading_speed_mps)
            if moving.any():
                moving_previous = previous_indices[moving]
                moving_current = current_indices[moving]
                bearings = np.asarray(
                    calculate_bearing(
                        longitude[moving_previous],
                        latitude[moving_previous],
                        longitude[moving_current],
                        latitude[moving_current],
                    ),
                    dtype=np.float64,
                )
                valid_bearings = np.isfinite(bearings)
                gps_heading_deg[moving_current[valid_bearings]] = bearings[valid_bearings]
                gps_heading_available[moving_current[valid_bearings]] = True

    valid_time = time_interval_valid & np.isfinite(time_delta) & (time_delta > 0)
    return {
        "time_delta_s": time_delta,
        "time_interval_valid": time_interval_valid,
        "time_gap_over_5s": (valid_time & (time_delta > 5.0)).astype(np.int8),
        "time_gap_over_60s": (valid_time & (time_delta > 60.0)).astype(np.int8),
        "gps_speed_mps": gps_speed_mps,
        "gps_speed_kmh": gps_speed_mps * 3.6,
        "gps_speed_available": gps_speed_available,
        "gps_movement_bearing_deg": gps_heading_deg,
        "gps_heading_available": gps_heading_available,
    }


def extract_candidate_features(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    candidates: pd.DataFrame,
    trip_id: str,
    minimum_heading_speed_mps: float = 0.5,
) -> pd.DataFrame:
    """Create finite, route-independent features for each candidate pair.

    Distances and lengths are metres; timestamps/deltas are seconds; headings
    and angular differences are degrees clockwise from north. Unavailable
    directional or temporal features use zero plus an explicit availability
    flag. No route or label data is accepted by this function.
    """
    if not np.isfinite(minimum_heading_speed_mps) or minimum_heading_speed_mps < 0:
        raise ValueError("minimum_heading_speed_mps must be finite and non-negative")
    expected_columns = {"gps_observation_index", "candidate_arc_id", "distance_m"}
    missing_columns = expected_columns.difference(candidates.columns)
    if missing_columns and not candidates.empty:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing_columns)}")
    if candidates.empty:
        return pd.DataFrame(columns=(*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS))

    observation_indices = _integer_values(candidates, "gps_observation_index")
    candidate_arc_ids = _integer_values(candidates, "candidate_arc_id")
    if (
        (observation_indices < 0).any()
        or (observation_indices >= len(track)).any()
    ):
        raise ValueError("Candidate GPS observation index is outside the track row range")
    if (candidate_arc_ids < 0).any() or (candidate_arc_ids >= len(arcs)).any():
        raise ValueError("Candidate arc ID is outside the .arcs row range")

    arc_endpoints = arcs.iloc[candidate_arc_ids][["from_node", "to_node"]].to_numpy(
        dtype=np.int64
    )
    if (
        (arc_endpoints < 0).any()
        or (arc_endpoints >= len(nodes)).any()
    ):
        raise ValueError("Candidate arc endpoint ID is outside the .nodes row range")
    start_node_ids = arc_endpoints[:, 0]
    end_node_ids = arc_endpoints[:, 1]

    longitude = _numeric_values(track, "longitude")
    latitude = _numeric_values(track, "latitude")
    valid_gps = _valid_coordinates(longitude, latitude)
    node_longitude = _numeric_values(nodes, "longitude")
    node_latitude = _numeric_values(nodes, "latitude")
    valid_nodes = _valid_coordinates(node_longitude, node_latitude)
    if not valid_gps[observation_indices].all():
        raise ValueError("Candidate rows refer to GPS observations with invalid coordinates")
    if not valid_nodes[start_node_ids].all() or not valid_nodes[end_node_ids].all():
        raise ValueError("Candidate rows refer to road arcs with invalid endpoint coordinates")

    candidate_distances = pd.to_numeric(
        candidates["distance_m"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(candidate_distances).all() or (candidate_distances < 0).any():
        raise ValueError("Candidate distances must be finite, non-negative metres")

    start_longitude = node_longitude[start_node_ids]
    start_latitude = node_latitude[start_node_ids]
    end_longitude = node_longitude[end_node_ids]
    end_latitude = node_latitude[end_node_ids]
    point_longitude = longitude[observation_indices]
    point_latitude = latitude[observation_indices]

    start_distances = np.asarray(
        calculate_distance(
            point_longitude,
            point_latitude,
            start_longitude,
            start_latitude,
        ),
        dtype=np.float64,
    )
    end_distances = np.asarray(
        calculate_distance(
            point_longitude,
            point_latitude,
            end_longitude,
            end_latitude,
        ),
        dtype=np.float64,
    )
    arc_lengths = np.asarray(
        calculate_distance(
            start_longitude,
            start_latitude,
            end_longitude,
            end_latitude,
        ),
        dtype=np.float64,
    )
    road_bearings = np.asarray(
        calculate_bearing(
            start_longitude,
            start_latitude,
            end_longitude,
            end_latitude,
        ),
        dtype=np.float64,
    )
    bearing_available = np.isfinite(road_bearings)
    road_bearings = np.where(bearing_available, road_bearings, 0.0)

    movement_features = _gps_movement_features(
        track, minimum_heading_speed_mps
    )
    gps_bearings = movement_features["gps_movement_bearing_deg"][
        observation_indices
    ]
    heading_available = movement_features["gps_heading_available"][
        observation_indices
    ].astype(bool)
    bearing_difference = np.zeros(len(candidates), dtype=np.float64)
    heading_consistency = np.zeros(len(candidates), dtype=np.float64)
    comparable_heading = heading_available & bearing_available
    if comparable_heading.any():
        difference = np.asarray(
            angular_difference(
                gps_bearings[comparable_heading],
                road_bearings[comparable_heading],
            ),
            dtype=np.float64,
        )
        bearing_difference[comparable_heading] = difference
        heading_consistency[comparable_heading] = np.cos(np.deg2rad(difference))

    feature_rows = pd.DataFrame(
        {
            "trip_id": str(trip_id),
            "gps_observation_index": observation_indices,
            "candidate_arc_id": candidate_arc_ids,
            "point_to_segment_distance_m": candidate_distances,
            "distance_to_start_node_m": start_distances,
            "distance_to_end_node_m": end_distances,
            "candidate_arc_length_m": arc_lengths,
            "candidate_road_bearing_deg": road_bearings,
            "candidate_bearing_available": bearing_available.astype(np.int8),
            "gps_movement_bearing_deg": gps_bearings,
            "gps_heading_available": heading_available.astype(np.int8),
            "bearing_difference_deg": bearing_difference,
            "heading_consistency": heading_consistency,
            "gps_speed_mps": movement_features["gps_speed_mps"][observation_indices],
            "gps_speed_kmh": movement_features["gps_speed_kmh"][observation_indices],
            "gps_speed_available": movement_features["gps_speed_available"][
                observation_indices
            ].astype(np.int8),
            "time_delta_s": movement_features["time_delta_s"][observation_indices],
            "time_interval_valid": movement_features["time_interval_valid"][
                observation_indices
            ].astype(np.int8),
            "time_gap_over_5s": movement_features["time_gap_over_5s"][
                observation_indices
            ],
            "time_gap_over_60s": movement_features["time_gap_over_60s"][
                observation_indices
            ],
        }
    )
    if not np.isfinite(feature_rows[list(_METRIC_FEATURES)].to_numpy()).all():
        raise ValueError("Feature extraction produced a non-finite numeric feature")
    return feature_rows.loc[:, [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]]
