"""Viterbi route reconstruction over per-observation road-arc candidates."""

from dataclasses import dataclass
import heapq
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.candidates import project_trip_coordinates
from src.geo import calculate_distance


FloatArray: TypeAlias = np.ndarray
_PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class ViterbiResult:
    """Classifier-only and HMM predictions, with within-segment continuity."""

    predictions: pd.DataFrame
    route_continuity: float


def _integer_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain integer values") from error
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{column} must contain integer values")
    return values.astype(np.int64)


def _coordinates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        longitude = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        latitude = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(
            dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Coordinate tables must contain longitude and latitude") from error
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )
    return longitude, latitude, valid


def _arc_endpoints(
    arcs: pd.DataFrame, node_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        starts = _integer_column(arcs, "from_node")
        ends = _integer_column(arcs, "to_node")
    except KeyError as error:
        raise ValueError("Arc table must contain from_node and to_node") from error
    valid = (
        (starts >= 0)
        & (starts < node_count)
        & (ends >= 0)
        & (ends < node_count)
    )
    return starts, ends, valid


def _projection_fractions(
    gps_xy: np.ndarray,
    node_xy: np.ndarray,
    observation_indices: np.ndarray,
    arc_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    arc_valid: np.ndarray,
    valid_gps: np.ndarray,
) -> np.ndarray:
    fractions = np.full(len(arc_ids), np.nan, dtype=np.float64)
    usable = valid_gps[observation_indices] & arc_valid[arc_ids]
    if not usable.any():
        return fractions
    rows = np.flatnonzero(usable)
    segment_starts = node_xy[starts[arc_ids[rows]]]
    vectors = node_xy[ends[arc_ids[rows]]] - segment_starts
    lengths_squared = np.sum(vectors * vectors, axis=1)
    offsets = gps_xy[observation_indices[rows]] - segment_starts
    fractions[rows] = np.clip(
        np.divide(
            np.sum(offsets * vectors, axis=1),
            lengths_squared,
            out=np.zeros(len(rows), dtype=np.float64),
            where=lengths_squared > 0.0,
        ),
        0.0,
        1.0,
    )
    return fractions


def _arc_lengths(
    longitude: np.ndarray,
    latitude: np.ndarray,
    valid_nodes: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    arc_valid: np.ndarray,
) -> np.ndarray:
    lengths = np.full(len(starts), np.nan, dtype=np.float64)
    if not len(valid_nodes):
        return lengths
    usable = arc_valid & valid_nodes[starts.clip(0, max(0, len(valid_nodes) - 1))] & (
        valid_nodes[ends.clip(0, max(0, len(valid_nodes) - 1))]
    )
    arc_ids = np.flatnonzero(usable)
    if arc_ids.size:
        lengths[arc_ids] = np.asarray(
            calculate_distance(
                longitude[starts[arc_ids]],
                latitude[starts[arc_ids]],
                longitude[ends[arc_ids]],
                latitude[ends[arc_ids]],
            ),
            dtype=np.float64,
        )
    return lengths


def _adjacency(
    starts: np.ndarray,
    ends: np.ndarray,
    lengths: np.ndarray,
    node_count: int,
) -> list[list[tuple[int, float]]]:
    graph: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for arc_id, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if np.isfinite(lengths[arc_id]):
            graph[int(start)].append((int(end), float(lengths[arc_id])))
    return graph


def _bounded_shortest_paths(
    graph: list[list[tuple[int, float]]],
    source: int,
    targets: set[int],
    distance_limit: float,
) -> dict[int, float]:
    unresolved = set(targets)
    distances: dict[int, float] = {}
    queue: list[tuple[float, int]] = [(0.0, source)]
    best: dict[int, float] = {source: 0.0}
    while queue and unresolved:
        distance, node = heapq.heappop(queue)
        if distance != best.get(node) or distance > distance_limit:
            continue
        if node in unresolved:
            distances[node] = distance
            unresolved.remove(node)
        for neighbour, edge_length in graph[node]:
            next_distance = distance + edge_length
            if next_distance <= distance_limit and next_distance < best.get(
                neighbour, float("inf")
            ):
                best[neighbour] = next_distance
                heapq.heappush(queue, (next_distance, neighbour))
    return distances


def _transition_matrix(
    previous_states: list[tuple[int, float, float]],
    next_states: list[tuple[int, float, float]],
    starts: np.ndarray,
    ends: np.ndarray,
    lengths: np.ndarray,
    graph: list[list[tuple[int, float]]],
    gps_distance_m: float,
    transition_sigma_m: float,
    relative_sigma: float,
) -> np.ndarray:
    sigma = max(transition_sigma_m, gps_distance_m * relative_sigma)
    plausibility_limit = gps_distance_m + 5.0 * sigma
    matrix = np.full(
        (len(previous_states), len(next_states)), -np.inf, dtype=np.float64
    )
    for previous_index, (previous_arc, _, previous_fraction) in enumerate(
        previous_states
    ):
        if not np.isfinite(previous_fraction) or not np.isfinite(lengths[previous_arc]):
            continue
        source = int(ends[previous_arc])
        target_nodes = {
            int(starts[next_arc])
            for next_arc, _, fraction in next_states
            if np.isfinite(fraction) and np.isfinite(lengths[next_arc])
        }
        if not target_nodes:
            continue
        paths = _bounded_shortest_paths(
            graph, source, target_nodes, plausibility_limit
        )
        previous_remainder = (1.0 - previous_fraction) * lengths[previous_arc]
        for next_index, (next_arc, _, next_fraction) in enumerate(next_states):
            if not np.isfinite(next_fraction) or not np.isfinite(lengths[next_arc]):
                continue
            if previous_arc == next_arc and next_fraction >= previous_fraction:
                network_distance = (
                    next_fraction - previous_fraction
                ) * lengths[previous_arc]
            else:
                path_distance = paths.get(int(starts[next_arc]))
                if path_distance is None:
                    continue
                network_distance = (
                    previous_remainder
                    + path_distance
                    + next_fraction * lengths[next_arc]
                )
            mismatch = network_distance - gps_distance_m
            if abs(mismatch) <= 5.0 * sigma:
                matrix[previous_index, next_index] = -0.5 * (mismatch / sigma) ** 2
    for row in matrix:
        finite = np.isfinite(row)
        if finite.any():
            maximum = float(row[finite].max())
            log_normalizer = maximum + np.log(np.exp(row[finite] - maximum).sum())
            row[finite] -= log_normalizer
    return matrix


def _decode_chunk(
    states: list[list[tuple[int, float, float]]],
    transition_matrices: list[np.ndarray],
) -> list[tuple[int, float]]:
    scores = np.log(
        np.clip(
            np.asarray([probability for _, probability, _ in states[0]]),
            _PROBABILITY_FLOOR,
            1.0,
        )
    )
    backpointers: list[np.ndarray] = []
    for position, transition in enumerate(transition_matrices, start=1):
        emissions = np.log(
            np.clip(
                np.asarray([probability for _, probability, _ in states[position]]),
                _PROBABILITY_FLOOR,
                1.0,
            )
        )
        candidates = scores[:, np.newaxis] + transition
        predecessors = np.argmax(candidates, axis=0)
        scores = candidates[predecessors, np.arange(len(emissions))] + emissions
        backpointers.append(predecessors)

    state_index = int(np.argmax(scores))
    decoded = [(states[-1][state_index][0], states[-1][state_index][1])]
    for position in range(len(backpointers) - 1, -1, -1):
        state_index = int(backpointers[position][state_index])
        decoded.append((states[position][state_index][0], states[position][state_index][1]))
    decoded.reverse()
    return decoded


def calculate_route_continuity(
    predicted_arc_ids: ArrayLike,
    arcs: pd.DataFrame,
    segment_ids: ArrayLike | None = None,
) -> float:
    """Return the connected-pair proportion, ignoring missing and split pairs."""
    predictions = np.asarray(predicted_arc_ids)
    if predictions.ndim != 1:
        raise ValueError("predicted_arc_ids must be one-dimensional")
    if not np.issubdtype(predictions.dtype, np.integer):
        numeric = predictions.astype(np.float64)
        if not np.isfinite(numeric).all() or not np.equal(
            numeric, np.floor(numeric)
        ).all():
            raise ValueError("predicted_arc_ids must contain integers")
        predictions = numeric.astype(np.int64)
    try:
        starts = _integer_column(arcs, "from_node")
        ends = _integer_column(arcs, "to_node")
    except KeyError as error:
        raise ValueError("Arc table must contain from_node and to_node") from error
    valid_arcs = (starts >= 0) & (ends >= 0)
    if ((predictions < -1) | (predictions >= len(arcs))).any():
        raise ValueError("predicted arc ID is outside the .arcs row range")
    if segment_ids is None:
        segments = np.zeros(len(predictions), dtype=np.int64)
    else:
        segments = np.asarray(segment_ids)
        if segments.ndim != 1 or len(segments) != len(predictions):
            raise ValueError("segment_ids must have one value per prediction")
    considered = (
        (predictions[:-1] >= 0)
        & (predictions[1:] >= 0)
        & (segments[:-1] == segments[1:])
    )
    pair_indices = np.flatnonzero(considered)
    if not len(pair_indices):
        return float("nan")
    previous = predictions[pair_indices].astype(np.int64)
    following = predictions[pair_indices + 1].astype(np.int64)
    valid_pairs = valid_arcs[previous] & valid_arcs[following]
    connected = np.zeros(len(pair_indices), dtype=bool)
    connected[valid_pairs] = (
        ends[previous[valid_pairs]] == starts[following[valid_pairs]]
    )
    return float(connected.sum() / len(pair_indices))


def decode_route(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_probabilities: ArrayLike | None = None,
    *,
    probability_column: str = "candidate_probability",
    max_gap_seconds: float = 60.0,
    transition_sigma_m: float = 50.0,
    relative_sigma: float = 0.25,
) -> ViterbiResult:
    """Decode candidate arc states for one trip using classifier probabilities.

    ``candidate_probabilities``, when supplied, must align row-for-row with
    ``candidates``. Otherwise ``probability_column`` is read from that table.
    Empty-candidate observations, invalid time/coordinate intervals, large time
    gaps, and transitions without a plausible directed network path split the
    Viterbi sequence; predictions are still returned for each available run.
    """
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be finite and positive")
    if not np.isfinite(transition_sigma_m) or transition_sigma_m <= 0:
        raise ValueError("transition_sigma_m must be finite and positive")
    if not np.isfinite(relative_sigma) or relative_sigma < 0:
        raise ValueError("relative_sigma must be finite and non-negative")
    required = {"gps_observation_index", "candidate_arc_id"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    observation_indices = _integer_column(candidates, "gps_observation_index")
    candidate_arc_ids = _integer_column(candidates, "candidate_arc_id")
    if candidate_probabilities is None:
        if probability_column not in candidates and not candidates.empty:
            raise ValueError(
                f"Candidate table is missing probability column '{probability_column}'"
            )
        candidate_probabilities = (
            candidates[probability_column]
            if probability_column in candidates
            else np.empty(0, dtype=np.float64)
        )
    try:
        probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Candidate probabilities must be numeric") from error
    if probabilities.ndim != 1 or len(probabilities) != len(candidates):
        raise ValueError("Candidate probabilities must align with candidate rows")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("Candidate probabilities must be finite values in [0, 1]")
    if (
        (observation_indices < 0).any()
        or (observation_indices >= len(track)).any()
    ):
        raise ValueError("Candidate observation index is outside the track row range")
    if (candidate_arc_ids < 0).any() or (candidate_arc_ids >= len(arcs)).any():
        raise ValueError("Candidate arc ID is outside the .arcs row range")
    if pd.DataFrame(
        {"observation": observation_indices, "arc": candidate_arc_ids}
    ).duplicated().any():
        raise ValueError("Candidate table contains duplicate arcs for an observation")

    try:
        timestamps = pd.to_numeric(track["timestamp"], errors="coerce").to_numpy(
            dtype=np.float64
        )
    except KeyError as error:
        raise ValueError("Track table must contain timestamps") from error
    longitude, latitude, valid_gps = _coordinates(track)
    node_longitude, node_latitude, valid_nodes = _coordinates(nodes)
    starts, ends, arc_valid = _arc_endpoints(arcs, len(nodes))
    projected = project_trip_coordinates(track, nodes)
    lengths = _arc_lengths(
        node_longitude, node_latitude, valid_nodes, starts, ends, arc_valid
    )
    graph = _adjacency(starts, ends, lengths, len(nodes))
    fractions = _projection_fractions(
        projected.gps_xy,
        projected.node_xy,
        observation_indices,
        candidate_arc_ids,
        starts,
        ends,
        arc_valid,
        valid_gps,
    )

    states_by_observation: list[list[tuple[int, float, float]]] = [
        [] for _ in range(len(track))
    ]
    for observation_index, arc_id, probability, fraction in zip(
        observation_indices, candidate_arc_ids, probabilities, fractions, strict=True
    ):
        states_by_observation[int(observation_index)].append(
            (int(arc_id), float(probability), float(fraction))
        )
    for states in states_by_observation:
        states.sort(key=lambda state: state[0])

    classifier_only = np.full(len(track), -1, dtype=np.int64)
    classifier_probability = np.full(len(track), np.nan, dtype=np.float64)
    for observation_index, states in enumerate(states_by_observation):
        if states:
            best_state = min(states, key=lambda state: (-state[1], state[0]))
            classifier_only[observation_index] = best_state[0]
            classifier_probability[observation_index] = best_state[1]

    hmm_predictions = np.full(len(track), -1, dtype=np.int64)
    hmm_probabilities = np.full(len(track), np.nan, dtype=np.float64)
    segment_ids = np.full(len(track), -1, dtype=np.int64)
    continuity_groups = np.full(len(track), -1, dtype=np.int64)
    statuses = np.full(len(track), "no_candidates", dtype=object)
    segment_counter = 0
    continuity_group_counter = 0
    current_continuity_group = -1

    def decode_run(observation_run: list[int]) -> None:
        nonlocal segment_counter
        chunk_start = 0
        transition_matrices: list[np.ndarray] = []

        def store_chunk(chunk_end: int) -> None:
            nonlocal segment_counter
            chunk = observation_run[chunk_start:chunk_end]
            decoded = _decode_chunk(
                [states_by_observation[index] for index in chunk],
                transition_matrices,
            )
            segment_id = segment_counter
            segment_counter += 1
            for observation_index, (arc_id, probability) in zip(
                chunk, decoded, strict=True
            ):
                hmm_predictions[observation_index] = arc_id
                hmm_probabilities[observation_index] = probability
                segment_ids[observation_index] = segment_id
                statuses[observation_index] = "decoded"

        for position in range(1, len(observation_run)):
            previous_index = observation_run[position - 1]
            next_index = observation_run[position]
            gps_distance = float(
                calculate_distance(
                    longitude[previous_index],
                    latitude[previous_index],
                    longitude[next_index],
                    latitude[next_index],
                )
            )
            matrix = _transition_matrix(
                states_by_observation[previous_index],
                states_by_observation[next_index],
                starts,
                ends,
                lengths,
                graph,
                gps_distance,
                transition_sigma_m,
                relative_sigma,
            )
            if not np.isfinite(matrix).any():
                store_chunk(position)
                chunk_start = position
                transition_matrices = []
            else:
                transition_matrices.append(matrix)
        store_chunk(len(observation_run))

    current_run: list[int] = []
    for observation_index, states in enumerate(states_by_observation):
        if not states:
            if current_run:
                decode_run(current_run)
                current_run = []
            current_continuity_group = -1
            continue
        if current_run:
            previous_index = current_run[-1]
            elapsed = timestamps[observation_index] - timestamps[previous_index]
            valid_interval = (
                np.isfinite(timestamps[observation_index])
                and np.isfinite(timestamps[previous_index])
                and 0.0 < elapsed <= max_gap_seconds
                and valid_gps[observation_index]
                and valid_gps[previous_index]
            )
            if observation_index != previous_index + 1 or not valid_interval:
                decode_run(current_run)
                current_run = []
                current_continuity_group = -1
        if not current_run:
            current_continuity_group = continuity_group_counter
            continuity_group_counter += 1
        continuity_groups[observation_index] = current_continuity_group
        current_run.append(observation_index)
    if current_run:
        decode_run(current_run)

    prediction_frame = pd.DataFrame(
        {
            "gps_observation_index": np.arange(len(track), dtype=np.int64),
            "classifier_only_arc_id": classifier_only,
            "classifier_only_probability": classifier_probability,
            "hmm_arc_id": hmm_predictions,
            "hmm_candidate_probability": hmm_probabilities,
            "decode_segment": segment_ids,
            "continuity_group": continuity_groups,
            "decode_status": statuses,
        }
    )
    continuity = calculate_route_continuity(hmm_predictions, arcs, continuity_groups)
    return ViterbiResult(prediction_frame, continuity)
