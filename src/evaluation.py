"""Evaluation helpers for route-aligned candidate coverage."""

import numpy as np
import pandas as pd

from src.candidates import CANDIDATE_COLUMNS, project_trip_coordinates
from src.geo import point_to_segment_distance


def build_route_reference_candidates(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    route: np.ndarray,
    trip_id: str,
    reference_radius_m: float = 50.0,
) -> pd.DataFrame:
    """Build uncapped, route-arc-only matches to establish reference labels.

    The route is used solely for training-label evaluation. All distances use a
    shared local metric CRS and only route arcs within the reference radius are
    eligible. This avoids defining reference labels from the capped candidate
    set being evaluated.
    """
    if not np.isfinite(reference_radius_m) or reference_radius_m <= 0:
        raise ValueError("reference_radius_m must be finite and positive")
    route_ids = np.asarray(route)
    if route_ids.ndim != 1:
        raise ValueError("route must be a one-dimensional sequence of arc IDs")
    if not np.issubdtype(route_ids.dtype, np.integer):
        try:
            numeric_route_ids = route_ids.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("route arc IDs must be integers") from error
        if not np.isfinite(numeric_route_ids).all() or not np.equal(
            numeric_route_ids, np.floor(numeric_route_ids)
        ).all():
            raise ValueError("route arc IDs must be integers")
        route_ids = numeric_route_ids.astype(np.int64)
    route_ids = route_ids.astype(np.int64, copy=False)
    if ((route_ids < 0) | (route_ids >= len(arcs))).any():
        raise ValueError("route contains an arc ID outside the .arcs row range")

    projected = project_trip_coordinates(track, nodes)
    unique_route_arc_ids = np.unique(route_ids)
    if unique_route_arc_ids.size == 0:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    endpoints = arcs.iloc[unique_route_arc_ids][["from_node", "to_node"]].to_numpy(
        dtype=np.int64
    )
    if ((endpoints < 0) | (endpoints >= len(nodes))).any():
        raise ValueError("route arc endpoint ID is outside the .nodes row range")
    valid_arc_endpoints = (
        projected.valid_nodes[endpoints[:, 0]]
        & projected.valid_nodes[endpoints[:, 1]]
    )
    unique_route_arc_ids = unique_route_arc_ids[valid_arc_endpoints]
    endpoints = endpoints[valid_arc_endpoints]
    rows: list[dict[str, int | float | str]] = []
    for observation_index, point_xy in enumerate(projected.gps_xy):
        if not projected.valid_gps[observation_index] or not unique_route_arc_ids.size:
            continue
        distances = np.asarray(
            point_to_segment_distance(
                point_xy,
                projected.node_xy[endpoints[:, 0]],
                projected.node_xy[endpoints[:, 1]],
            ),
            dtype=np.float64,
        )
        selected = np.flatnonzero(distances <= reference_radius_m)
        order = selected[np.lexsort((unique_route_arc_ids[selected], distances[selected]))]
        for rank, match_index in enumerate(order, start=1):
            rows.append(
                {
                    "trip_id": str(trip_id),
                    "gps_observation_index": int(observation_index),
                    "candidate_arc_id": int(unique_route_arc_ids[match_index]),
                    "distance_m": float(distances[match_index]),
                    "candidate_rank": rank,
                }
            )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def candidate_recall_summary(
    reference_labels: pd.DataFrame,
    candidate_sets: dict[float, pd.DataFrame],
    observation_count: int,
) -> pd.DataFrame:
    """Measure capped candidate recall against labelled reference observations."""
    labelled = reference_labels.loc[
        reference_labels["label_status"] == "labelled",
        ["gps_observation_index", "label_arc_id"],
    ].copy()
    labelled["gps_observation_index"] = labelled[
        "gps_observation_index"
    ].astype(np.int64)
    labelled["label_arc_id"] = labelled["label_arc_id"].astype(np.int64)
    denominator = len(labelled)
    summary_rows: list[dict[str, int | float]] = []
    for radius_m, candidates in sorted(candidate_sets.items()):
        pair_keys = set(
            zip(
                pd.to_numeric(candidates.get("gps_observation_index", pd.Series(dtype=int))),
                pd.to_numeric(candidates.get("candidate_arc_id", pd.Series(dtype=int))),
            )
        )
        hits = sum(
            (int(row.gps_observation_index), int(row.label_arc_id)) in pair_keys
            for row in labelled.itertuples(index=False)
        )
        observations_with_candidates = (
            int(candidates["gps_observation_index"].nunique())
            if not candidates.empty
            else 0
        )
        summary_rows.append(
            {
                "search_radius_m": float(radius_m),
                "reference_label_count": int(denominator),
                "candidate_hits": int(hits),
                "candidate_recall": float(hits / denominator) if denominator else np.nan,
                "label_coverage": (
                    float(denominator / observation_count) if observation_count else np.nan
                ),
                "observations_with_candidates": observations_with_candidates,
                "observations_without_candidates": max(
                    0, observation_count - observations_with_candidates
                ),
            }
        )
    return pd.DataFrame(summary_rows)
