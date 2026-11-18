"""Spatial candidate generation for directed road arcs."""

from collections import defaultdict
from dataclasses import dataclass
from math import floor
from typing import TypeAlias

import numpy as np
import pandas as pd

from src.geo import point_to_segment_distance, project_coordinates


FloatArray: TypeAlias = np.ndarray
CANDIDATE_COLUMNS = (
    "trip_id",
    "gps_observation_index",
    "candidate_arc_id",
    "distance_m",
    "candidate_rank",
)
OBSERVATION_COLUMNS = (
    "trip_id",
    "gps_observation_index",
    "has_valid_coordinates",
    "candidate_count",
    "nearby_arc_count",
    "search_radius_m",
)


@dataclass(frozen=True)
class ProjectedTripCoordinates:
    """GPS fixes and network nodes in one shared local metric CRS."""

    gps_xy: FloatArray
    node_xy: FloatArray
    valid_gps: np.ndarray
    valid_nodes: np.ndarray


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Candidate pairs, per-observation counts, and invalid arc IDs."""

    candidates: pd.DataFrame
    observations: pd.DataFrame
    invalid_arc_ids: tuple[int, ...]


def _coordinate_arrays(frame: pd.DataFrame) -> tuple[FloatArray, FloatArray, np.ndarray]:
    longitude = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    latitude = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )
    return longitude, latitude, valid


def project_trip_coordinates(
    track: pd.DataFrame, nodes: pd.DataFrame
) -> ProjectedTripCoordinates:
    """Project GPS fixes and road nodes together to ensure a shared CRS.

    Longitude/latitude inputs are in degrees. Returned x/y coordinates are in
    metres in one azimuthal-equidistant CRS chosen for the trip.
    """
    gps_longitude, gps_latitude, valid_gps = _coordinate_arrays(track)
    node_longitude, node_latitude, valid_nodes = _coordinate_arrays(nodes)
    gps_xy = np.full((len(track), 2), np.nan, dtype=np.float64)
    node_xy = np.full((len(nodes), 2), np.nan, dtype=np.float64)
    if not valid_gps.any() and not valid_nodes.any():
        return ProjectedTripCoordinates(gps_xy, node_xy, valid_gps, valid_nodes)

    all_longitudes = np.concatenate(
        (gps_longitude[valid_gps], node_longitude[valid_nodes])
    )
    all_latitudes = np.concatenate((gps_latitude[valid_gps], node_latitude[valid_nodes]))
    x_coordinates, y_coordinates = project_coordinates(all_longitudes, all_latitudes)
    gps_count = int(valid_gps.sum())
    gps_xy[valid_gps] = np.column_stack(
        (x_coordinates[:gps_count], y_coordinates[:gps_count])
    )
    node_xy[valid_nodes] = np.column_stack(
        (x_coordinates[gps_count:], y_coordinates[gps_count:])
    )
    return ProjectedTripCoordinates(gps_xy, node_xy, valid_gps, valid_nodes)


class SegmentSpatialIndex:
    """Uniform-grid index for exact point-to-segment searches in metres."""

    _MAX_CELLS_PER_SEGMENT = 128

    def __init__(
        self,
        starts_xy: FloatArray,
        ends_xy: FloatArray,
        item_ids: np.ndarray,
        cell_size_m: float,
    ) -> None:
        self.starts_xy = np.asarray(starts_xy, dtype=np.float64)
        self.ends_xy = np.asarray(ends_xy, dtype=np.float64)
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.cell_size_m = float(cell_size_m)
        self.minimum_x = np.minimum(self.starts_xy[:, 0], self.ends_xy[:, 0])
        self.maximum_x = np.maximum(self.starts_xy[:, 0], self.ends_xy[:, 0])
        self.minimum_y = np.minimum(self.starts_xy[:, 1], self.ends_xy[:, 1])
        self.maximum_y = np.maximum(self.starts_xy[:, 1], self.ends_xy[:, 1])
        self._grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._large_segments: list[int] = []

        for segment_index in range(len(self.item_ids)):
            minimum_cell_x = floor(self.minimum_x[segment_index] / self.cell_size_m)
            maximum_cell_x = floor(self.maximum_x[segment_index] / self.cell_size_m)
            minimum_cell_y = floor(self.minimum_y[segment_index] / self.cell_size_m)
            maximum_cell_y = floor(self.maximum_y[segment_index] / self.cell_size_m)
            covered_cells = (
                (maximum_cell_x - minimum_cell_x + 1)
                * (maximum_cell_y - minimum_cell_y + 1)
            )
            if covered_cells > self._MAX_CELLS_PER_SEGMENT:
                self._large_segments.append(segment_index)
                continue
            for cell_x in range(minimum_cell_x, maximum_cell_x + 1):
                for cell_y in range(minimum_cell_y, maximum_cell_y + 1):
                    self._grid[(cell_x, cell_y)].append(segment_index)

    def query(self, point_xy: FloatArray, radius_m: float) -> tuple[np.ndarray, FloatArray]:
        """Return indexed item IDs and exact distances within ``radius_m``."""
        minimum_cell_x = floor((point_xy[0] - radius_m) / self.cell_size_m)
        maximum_cell_x = floor((point_xy[0] + radius_m) / self.cell_size_m)
        minimum_cell_y = floor((point_xy[1] - radius_m) / self.cell_size_m)
        maximum_cell_y = floor((point_xy[1] + radius_m) / self.cell_size_m)
        possible_segments: set[int] = set()
        for cell_x in range(minimum_cell_x, maximum_cell_x + 1):
            for cell_y in range(minimum_cell_y, maximum_cell_y + 1):
                possible_segments.update(self._grid.get((cell_x, cell_y), ()))

        if self._large_segments:
            large_indices = np.asarray(self._large_segments, dtype=np.int64)
            intersects_query = (
                (self.minimum_x[large_indices] <= point_xy[0] + radius_m)
                & (self.maximum_x[large_indices] >= point_xy[0] - radius_m)
                & (self.minimum_y[large_indices] <= point_xy[1] + radius_m)
                & (self.maximum_y[large_indices] >= point_xy[1] - radius_m)
            )
            possible_segments.update(large_indices[intersects_query].tolist())
        if not possible_segments:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        segment_indices = np.fromiter(
            sorted(possible_segments), dtype=np.int64, count=len(possible_segments)
        )
        distances = np.asarray(
            point_to_segment_distance(
                point_xy,
                self.starts_xy[segment_indices],
                self.ends_xy[segment_indices],
            ),
            dtype=np.float64,
        )
        within_radius = distances <= radius_m
        return self.item_ids[segment_indices[within_radius]], distances[within_radius]


def _valid_arc_segments(
    arcs: pd.DataFrame, projected: ProjectedTripCoordinates
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from_nodes = pd.to_numeric(arcs["from_node"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    to_nodes = pd.to_numeric(arcs["to_node"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    integer_endpoints = (
        np.isfinite(from_nodes)
        & np.isfinite(to_nodes)
        & (from_nodes == np.floor(from_nodes))
        & (to_nodes == np.floor(to_nodes))
    )
    in_range = (
        integer_endpoints
        & (from_nodes >= 0)
        & (to_nodes >= 0)
        & (from_nodes < len(projected.node_xy))
        & (to_nodes < len(projected.node_xy))
    )
    valid = np.zeros(len(arcs), dtype=bool)
    valid_indices = np.flatnonzero(in_range)
    if valid_indices.size:
        start_ids = from_nodes[valid_indices].astype(np.int64)
        end_ids = to_nodes[valid_indices].astype(np.int64)
        valid[valid_indices] = (
            projected.valid_nodes[start_ids]
            & projected.valid_nodes[end_ids]
        )
    arc_ids = np.flatnonzero(valid).astype(np.int64)
    if arc_ids.size == 0:
        return arc_ids, np.empty((0, 2)), np.empty((0, 2))
    endpoints = arcs.iloc[arc_ids][["from_node", "to_node"]].to_numpy(dtype=np.int64)
    return arc_ids, projected.node_xy[endpoints[:, 0]], projected.node_xy[endpoints[:, 1]]


def generate_candidates(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    trip_id: str,
    search_radius_m: float = 50.0,
    max_candidates: int = 8,
    grid_cell_size_m: float | None = None,
) -> CandidateGenerationResult:
    """Find nearby directed arc candidates for each GPS observation.

    Candidate distances are exact point-to-segment distances in a shared local
    metric CRS. Arc IDs remain the original zero-based `.arcs` row positions;
    reverse-direction arcs remain distinct candidates.
    """
    if not np.isfinite(search_radius_m) or search_radius_m <= 0:
        raise ValueError("search_radius_m must be a finite, positive number")
    if isinstance(max_candidates, bool) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if int(max_candidates) != max_candidates:
        raise ValueError("max_candidates must be a positive integer")
    cell_size = (
        float(grid_cell_size_m)
        if grid_cell_size_m is not None
        else float(search_radius_m)
    )
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("grid_cell_size_m must be a finite, positive number")

    projected = project_trip_coordinates(track, nodes)
    arc_ids, starts_xy, ends_xy = _valid_arc_segments(arcs, projected)
    invalid_arc_ids = tuple(
        int(arc_id) for arc_id in np.flatnonzero(
            ~np.isin(np.arange(len(arcs)), arc_ids)
        )
    )
    if arc_ids.size:
        spatial_index = SegmentSpatialIndex(starts_xy, ends_xy, arc_ids, cell_size)
    else:
        spatial_index = None

    candidate_rows: list[dict[str, int | float | str]] = []
    observation_rows: list[dict[str, int | float | str | bool]] = []
    for observation_index, point_xy in enumerate(projected.gps_xy):
        matching_arc_ids = np.empty(0, dtype=np.int64)
        distances = np.empty(0, dtype=np.float64)
        if projected.valid_gps[observation_index] and spatial_index is not None:
            matching_arc_ids, distances = spatial_index.query(
                point_xy, float(search_radius_m)
            )
        order = np.lexsort((matching_arc_ids, distances))
        nearby_count = int(order.size)
        selected = order[: int(max_candidates)]
        for rank, match_index in enumerate(selected, start=1):
            candidate_rows.append(
                {
                    "trip_id": str(trip_id),
                    "gps_observation_index": int(observation_index),
                    "candidate_arc_id": int(matching_arc_ids[match_index]),
                    "distance_m": float(distances[match_index]),
                    "candidate_rank": rank,
                }
            )
        observation_rows.append(
            {
                "trip_id": str(trip_id),
                "gps_observation_index": int(observation_index),
                "has_valid_coordinates": bool(projected.valid_gps[observation_index]),
                "candidate_count": int(len(selected)),
                "nearby_arc_count": nearby_count,
                "search_radius_m": float(search_radius_m),
            }
        )
    return CandidateGenerationResult(
        candidates=pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS),
        observations=pd.DataFrame(observation_rows, columns=OBSERVATION_COLUMNS),
        invalid_arc_ids=invalid_arc_ids,
    )
