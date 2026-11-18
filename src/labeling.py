"""Route-order-aware, conservative labels for training candidate arcs."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


LABEL_COLUMNS = (
    "trip_id",
    "gps_observation_index",
    "label_arc_id",
    "label_route_position",
    "label_status",
    "alignment_component",
)


@dataclass(frozen=True)
class _MatchNode:
    observation_index: int
    route_position: int
    previous: "_MatchNode | None"


@dataclass(frozen=True)
class _ComponentNode:
    observation_index: int
    component_index: int
    previous: "_ComponentNode | None"


@dataclass(frozen=True)
class _State:
    last_route_position: int
    match_count: int
    total_distance_m: float
    path: _MatchNode | _ComponentNode | None


def split_route_components(route: np.ndarray, arcs: pd.DataFrame) -> list[np.ndarray]:
    """Split route-position indices where consecutive directed arcs disconnect."""
    route_ids = np.asarray(route)
    if route_ids.ndim != 1:
        raise ValueError("route must be a one-dimensional sequence of arc IDs")
    if not np.issubdtype(route_ids.dtype, np.integer):
        if not np.isfinite(route_ids).all() or not np.equal(
            route_ids, np.floor(route_ids)
        ).all():
            raise ValueError("route arc IDs must be integers")
        route_ids = route_ids.astype(np.int64)
    if ((route_ids < 0) | (route_ids >= len(arcs))).any():
        raise ValueError("route contains an arc ID outside the .arcs row range")
    if route_ids.size == 0:
        return []

    route_endpoints = arcs.iloc[route_ids][["from_node", "to_node"]].to_numpy(
        dtype=np.int64
    )
    breaks = np.flatnonzero(route_endpoints[:-1, 1] != route_endpoints[1:, 0]) + 1
    return [
        component.astype(np.int64)
        for component in np.split(np.arange(route_ids.size, dtype=np.int64), breaks)
        if component.size
    ]


def _state_is_better(candidate: _State, incumbent: _State) -> bool:
    if candidate.match_count != incumbent.match_count:
        return candidate.match_count > incumbent.match_count
    return candidate.total_distance_m < incumbent.total_distance_m


def _prune_frontier(states: list[_State]) -> list[_State]:
    best_by_position: dict[int, _State] = {}
    for state in states:
        incumbent = best_by_position.get(state.last_route_position)
        if incumbent is None or _state_is_better(state, incumbent):
            best_by_position[state.last_route_position] = state

    frontier: list[_State] = []
    best_count = -1
    best_cost = float("inf")
    for position in sorted(best_by_position):
        state = best_by_position[position]
        if state.match_count > best_count or (
            state.match_count == best_count and state.total_distance_m < best_cost
        ):
            frontier.append(state)
            if state.match_count > best_count:
                best_count = state.match_count
                best_cost = state.total_distance_m
            else:
                best_cost = state.total_distance_m
    return frontier


def _align_component(
    observation_count: int,
    route_positions: np.ndarray,
    route_ids: np.ndarray,
    candidates: pd.DataFrame,
    ambiguity_margin_m: float,
) -> tuple[dict[int, tuple[int, int]], set[int], set[int]]:
    """Return observation matches, ambiguous observations, and route candidate hits."""
    arc_positions: dict[int, list[int]] = defaultdict(list)
    for local_position, route_position in enumerate(route_positions):
        arc_positions[int(route_ids[route_position])].append(local_position)

    candidates_by_observation: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in candidates.itertuples(index=False):
        arc_id = int(row.candidate_arc_id)
        if arc_id in arc_positions:
            candidates_by_observation[int(row.gps_observation_index)].append(
                (arc_id, float(row.distance_m))
            )

    ambiguous_observations: set[int] = set()
    route_candidate_hits: set[int] = set(candidates_by_observation)
    for observation_index, options in candidates_by_observation.items():
        best_distance_by_arc: dict[int, float] = {}
        for arc_id, distance_m in options:
            best_distance_by_arc[arc_id] = min(
                distance_m, best_distance_by_arc.get(arc_id, float("inf"))
            )
        ordered_distances = sorted(best_distance_by_arc.values())
        if len(ordered_distances) > 1 and (
            ordered_distances[1] - ordered_distances[0] <= ambiguity_margin_m
        ):
            ambiguous_observations.add(observation_index)

    frontier = [_State(-1, 0, 0.0, None)]
    for observation_index in range(observation_count):
        next_states = list(frontier)
        for arc_id, distance_m in candidates_by_observation.get(
            observation_index, ()
        ):
            for local_position in arc_positions[arc_id]:
                predecessor = max(
                    (
                        state
                        for state in frontier
                        if state.last_route_position <= local_position
                    ),
                    key=lambda state: (
                        state.match_count,
                        -state.total_distance_m,
                    ),
                    default=None,
                )
                if predecessor is None:
                    continue
                next_states.append(
                    _State(
                        local_position,
                        predecessor.match_count + 1,
                        predecessor.total_distance_m + distance_m,
                        _MatchNode(
                            observation_index,
                            local_position,
                            predecessor.path,
                        ),
                    )
                )
        frontier = _prune_frontier(next_states)

    best_state = max(
        frontier,
        key=lambda state: (state.match_count, -state.total_distance_m),
    )
    matches: dict[int, tuple[int, int]] = {}
    path_node = best_state.path
    while path_node is not None:
        if not isinstance(path_node, _MatchNode):
            raise RuntimeError("Unexpected route alignment backpointer")
        original_route_position = int(route_positions[path_node.route_position])
        matches[path_node.observation_index] = (
            int(route_ids[original_route_position]),
            original_route_position,
        )
        path_node = path_node.previous
    return matches, ambiguous_observations, route_candidate_hits


def _ordered_component_selection(
    component_matches: list[dict[int, tuple[int, int]]],
    candidates: pd.DataFrame,
    observation_count: int,
) -> dict[int, int]:
    """Choose component matches in route order without crossing breaks."""
    distances: dict[tuple[int, int], float] = {}
    if not candidates.empty:
        for row in candidates.itertuples(index=False):
            key = (int(row.gps_observation_index), int(row.candidate_arc_id))
            distances[key] = min(
                float(row.distance_m), distances.get(key, float("inf"))
            )

    frontier = [_State(-1, 0, 0.0, None)]
    for observation_index in range(observation_count):
        next_states = list(frontier)
        options = []
        for component_index, matches in enumerate(component_matches):
            match = matches.get(observation_index)
            if match is not None:
                options.append(
                    (
                        component_index,
                        match[0],
                        match[1],
                        distances.get((observation_index, match[0]), 0.0),
                    )
                )
        for component_index, arc_id, route_position, distance_m in options:
            predecessor = max(
                (
                    state
                    for state in frontier
                    if state.last_route_position <= component_index
                ),
                key=lambda state: (state.match_count, -state.total_distance_m),
                default=None,
            )
            if predecessor is None:
                continue
            next_states.append(
                _State(
                    component_index,
                    predecessor.match_count + 1,
                    predecessor.total_distance_m + distance_m,
                    _ComponentNode(
                        observation_index,
                        component_index,
                        predecessor.path,
                    ),
                )
            )
        frontier = _prune_frontier(next_states)

    best_state = max(
        frontier,
        key=lambda state: (state.match_count, -state.total_distance_m),
    )
    selected: dict[int, int] = {}
    path_node = best_state.path
    while path_node is not None:
        if isinstance(path_node, _ComponentNode):
            selected[path_node.observation_index] = path_node.component_index
        path_node = path_node.previous
    return selected


def align_route_labels(
    route: np.ndarray,
    arcs: pd.DataFrame,
    candidates: pd.DataFrame,
    observation_count: int,
    trip_id: str,
    ambiguity_margin_m: float = 5.0,
) -> pd.DataFrame:
    """Align candidate arcs monotonically to route components.

    The route is used only to create training labels. Repeated observations may
    match the same route position; route positions can be skipped, but never
    traversed backwards. Disconnected route components are aligned separately.
    Ambiguous, conflicting, or order-infeasible observations remain unlabelled.
    """
    if observation_count < 0:
        raise ValueError("observation_count must be non-negative")
    if not np.isfinite(ambiguity_margin_m) or ambiguity_margin_m < 0:
        raise ValueError("ambiguity_margin_m must be finite and non-negative")
    required = {"gps_observation_index", "candidate_arc_id", "distance_m"}
    missing = required.difference(candidates.columns)
    if missing and not candidates.empty:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if not candidates.empty:
        indices = pd.to_numeric(candidates["gps_observation_index"], errors="raise")
        arc_ids = pd.to_numeric(candidates["candidate_arc_id"], errors="raise")
        distances = pd.to_numeric(candidates["distance_m"], errors="raise")
        if (
            (indices < 0).any()
            or (indices >= observation_count).any()
            or (indices != np.floor(indices)).any()
        ):
            raise ValueError("Candidate GPS observation index is invalid")
        if (arc_ids < 0).any() or (arc_ids != np.floor(arc_ids)).any():
            raise ValueError("Candidate arc IDs must be non-negative integers")
        if (arc_ids >= len(arcs)).any():
            raise ValueError("Candidate arc ID is outside the .arcs row range")
        if not np.isfinite(distances.to_numpy(dtype=np.float64)).all() or (
            distances < 0
        ).any():
            raise ValueError("Candidate distances must be finite, non-negative metres")

    raw_route_ids = np.asarray(route)
    if raw_route_ids.ndim != 1:
        raise ValueError("route must be a one-dimensional sequence of arc IDs")
    if not np.issubdtype(raw_route_ids.dtype, np.integer):
        try:
            numeric_route_ids = raw_route_ids.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("route arc IDs must be integers") from error
        if not np.isfinite(numeric_route_ids).all() or not np.equal(
            numeric_route_ids, np.floor(numeric_route_ids)
        ).all():
            raise ValueError("route arc IDs must be integers")
        raw_route_ids = numeric_route_ids.astype(np.int64)
    route_ids = raw_route_ids.astype(np.int64, copy=False)
    components = split_route_components(route_ids, arcs)
    component_matches: list[dict[int, tuple[int, int]]] = []
    ambiguous_observations: set[int] = set()
    route_candidate_hits: set[int] = set()
    for component_index, route_positions in enumerate(components):
        component_arc_ids = set(route_ids[route_positions].tolist())
        component_candidates = candidates.loc[
            candidates["candidate_arc_id"].isin(component_arc_ids)
        ] if not candidates.empty else candidates
        matches, ambiguous, hits = _align_component(
            observation_count,
            route_positions,
            route_ids,
            component_candidates,
            ambiguity_margin_m,
        )
        component_matches.append(matches)
        ambiguous_observations.update(ambiguous)
        route_candidate_hits.update(hits)

    selected_components = _ordered_component_selection(
        component_matches, candidates, observation_count
    )
    label_arc_ids = np.full(observation_count, -1, dtype=np.int64)
    label_route_positions = np.full(observation_count, -1, dtype=np.int64)
    alignment_components = np.full(observation_count, -1, dtype=np.int64)
    label_status = np.full(observation_count, "no_candidate", dtype=object)
    candidate_observations = set(
        pd.to_numeric(candidates["gps_observation_index"], errors="raise").astype(int)
    ) if not candidates.empty else set()
    for observation_index in range(observation_count):
        matching_components = [
            component_index
            for component_index, matches in enumerate(component_matches)
            if observation_index in matches
        ]
        if len(matching_components) > 1:
            label_status[observation_index] = "component_conflict"
            continue
        if not matching_components:
            if observation_index in ambiguous_observations:
                label_status[observation_index] = "ambiguous"
            elif observation_index in route_candidate_hits:
                label_status[observation_index] = "infeasible"
            elif observation_index in candidate_observations:
                label_status[observation_index] = "not_on_route"
            continue
        component_index = selected_components.get(observation_index)
        if component_index is None:
            label_status[observation_index] = "infeasible"
            continue
        arc_id, route_position = component_matches[component_index][observation_index]
        if observation_index in ambiguous_observations:
            label_status[observation_index] = "ambiguous"
            continue
        label_arc_ids[observation_index] = arc_id
        label_route_positions[observation_index] = route_position
        alignment_components[observation_index] = component_index
        label_status[observation_index] = "labelled"

    return pd.DataFrame(
        {
            "trip_id": str(trip_id),
            "gps_observation_index": np.arange(observation_count, dtype=np.int64),
            "label_arc_id": pd.array(
                [None if arc_id < 0 else int(arc_id) for arc_id in label_arc_ids],
                dtype="Int64",
            ),
            "label_route_position": pd.array(
                [
                    None if position < 0 else int(position)
                    for position in label_route_positions
                ],
                dtype="Int64",
            ),
            "label_status": label_status,
            "alignment_component": pd.array(
                [
                    None if component < 0 else int(component)
                    for component in alignment_components
                ],
                dtype="Int64",
            ),
        },
        columns=LABEL_COLUMNS,
    )
