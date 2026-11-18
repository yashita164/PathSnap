"""Prediction service: trip -> candidates -> features -> probabilities -> decoding.

Data policy enforced here:
  * Only ``settings.split_directory`` (student_test) is ever searched.
  * Only .track, .nodes and .arcs files are opened. ``.route`` files and
    ``src.io.load_trip`` (which reads them) are never used.
  * Existing candidate generation and feature extraction are reused unchanged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from api.config import Settings
from api.decoding import UNMATCHED, DecodeInput, DecodeResult, get_decoder
from api.errors import (
    ApiError,
    CandidateGenerationError,
    DecoderError,
    FeatureExtractionError,
    InvalidTripIdError,
    TripDataIncompleteError,
    TripDataInvalidError,
    TripNotFoundError,
)
from api.model_loader import ProbabilityClassifier
from api.schemas import (
    ArcFeature,
    ArcProperties,
    GeometryCollection,
    GpsPoint,
    LineStringGeometry,
    ObservationMatch,
    PredictMetrics,
    PredictRequest,
    PredictResponse,
    TimingsMs,
)
from src.candidates import generate_candidates
from src.features import FEATURE_COLUMNS, extract_candidate_features
from src.io import load_arcs, load_nodes, load_track

logger = logging.getLogger(__name__)

TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# --------------------------------------------------------------------- trips
def discover_trips(settings: Settings) -> dict[str, Path]:
    """Map trip_id -> directory for trips under the development split only."""
    split_directory = settings.split_directory
    if not split_directory.is_dir():
        return {}
    trips: dict[str, Path] = {}
    for track_path in sorted(split_directory.rglob("*.track")):
        trip_id = track_path.parent.name
        if track_path.stem == trip_id and TRIP_ID_PATTERN.fullmatch(trip_id):
            trips.setdefault(trip_id, track_path.parent)
    return trips


def list_trip_ids(settings: Settings) -> list[str]:
    return sorted(discover_trips(settings))


def resolve_trip(settings: Settings, trip_id: str) -> Path:
    """Validate the ID, then require it to be a discovered development trip.

    Looking the ID up in the discovered set (rather than joining it onto a path)
    also rules out path traversal.
    """
    if not TRIP_ID_PATTERN.fullmatch(trip_id):
        raise InvalidTripIdError(
            "Trip IDs may contain only letters, digits, '.', '_' and '-'."
        )
    directory = discover_trips(settings).get(trip_id)
    if directory is None:
        raise TripNotFoundError(f"Trip '{trip_id}' was not found.")
    return directory


def load_trip_inputs(
    directory: Path, trip_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load track, nodes and arcs (never the route) with client-safe errors."""
    loaded: dict[str, pd.DataFrame] = {}
    for extension, loader in (
        ("track", load_track),
        ("nodes", load_nodes),
        ("arcs", load_arcs),
    ):
        path = directory / f"{trip_id}.{extension}"
        if not path.is_file():
            raise TripDataIncompleteError(
                f"Trip '{trip_id}' is missing its required .{extension} file."
            )
        try:
            loaded[extension] = loader(path)
        except (OSError, ValueError) as error:
            logger.warning("Invalid .%s file for trip %s: %s", extension, trip_id, error)
            raise TripDataInvalidError(
                f"Trip '{trip_id}' has an unreadable or invalid .{extension} file."
            ) from error
    return loaded["track"], loaded["nodes"], loaded["arcs"]


# ------------------------------------------------------------------- options
@dataclass(frozen=True)
class PredictOptions:
    decoder: str
    search_radius_m: float
    max_candidates: int


def resolve_options(request: PredictRequest | None, settings: Settings) -> PredictOptions:
    request = request or PredictRequest()
    return PredictOptions(
        decoder=request.decoder or settings.decoder,
        search_radius_m=(
            request.search_radius_m
            if request.search_radius_m is not None
            else settings.search_radius_m
        ),
        max_candidates=(
            request.max_candidates
            if request.max_candidates is not None
            else settings.max_candidates
        ),
    )


# ------------------------------------------------------------------ geometry
def _collapse_consecutive(arc_ids: np.ndarray) -> np.ndarray:
    if arc_ids.size == 0:
        return arc_ids
    keep = np.concatenate(([True], arc_ids[1:] != arc_ids[:-1]))
    return arc_ids[keep]


def build_geometry(
    sequence: np.ndarray, nodes: pd.DataFrame, arcs: pd.DataFrame
) -> GeometryCollection:
    endpoints = arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    lonlat = nodes[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    features: list[ArcFeature] = []
    for position, arc_id in enumerate(sequence.tolist()):
        start, end = endpoints[arc_id]
        if not (0 <= start < len(lonlat) and 0 <= end < len(lonlat)):
            raise DecoderError("The decoder selected an arc with invalid endpoints.")
        features.append(
            ArcFeature(
                geometry=LineStringGeometry(
                    coordinates=[
                        [float(lonlat[start, 0]), float(lonlat[start, 1])],
                        [float(lonlat[end, 0]), float(lonlat[end, 1])],
                    ]
                ),
                properties=ArcProperties(arc_id=int(arc_id), sequence_index=position),
            )
        )
    return GeometryCollection(features=features)


def _valid_gps_points(track: pd.DataFrame) -> list[GpsPoint]:
    longitude = pd.to_numeric(track["longitude"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    latitude = pd.to_numeric(track["latitude"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )
    return [
        GpsPoint(
            gps_observation_index=int(index),
            longitude=float(longitude[index]),
            latitude=float(latitude[index]),
        )
        for index in np.flatnonzero(valid)
    ]


# ------------------------------------------------------------------ pipeline
def _validated_decode_result(
    result: DecodeResult, n_observations: int, n_arcs: int
) -> tuple[np.ndarray, np.ndarray]:
    arc_ids = np.asarray(result.arc_ids)
    scores = np.asarray(result.scores, dtype=np.float64)
    if arc_ids.shape != (n_observations,) or scores.shape != (n_observations,):
        raise DecoderError("The decoder returned results of the wrong length.")
    if not np.issubdtype(arc_ids.dtype, np.integer):
        raise DecoderError("The decoder must return integer arc IDs.")
    if ((arc_ids < UNMATCHED) | (arc_ids >= n_arcs)).any():
        raise DecoderError("The decoder returned arc IDs outside the network.")
    return arc_ids.astype(np.int64), scores


def predict_trip(
    trip_id: str,
    settings: Settings,
    load_classifier: Callable[[Settings], ProbabilityClassifier],
    options: PredictOptions,
) -> PredictResponse:
    started = perf_counter()
    directory = resolve_trip(settings, trip_id)  # 404 / 422 before anything heavy
    decoder = get_decoder(options.decoder)  # 422 unknown decoder
    classifier = load_classifier(settings)  # 503 model unavailable

    track, nodes, arcs = load_trip_inputs(directory, trip_id)
    n_observations = len(track)
    t_loaded = perf_counter()

    try:
        generated = generate_candidates(
            track,
            nodes,
            arcs,
            trip_id,
            search_radius_m=options.search_radius_m,
            max_candidates=options.max_candidates,
        )
    except (ValueError, OSError) as error:
        logger.warning("Candidate generation failed for %s: %s", trip_id, error)
        raise CandidateGenerationError(
            f"Candidates could not be generated for trip '{trip_id}'."
        ) from error
    candidates = generated.candidates
    t_candidates = perf_counter()

    if candidates.empty:
        decode_result = DecodeResult(
            np.full(n_observations, UNMATCHED, dtype=np.int64),
            np.full(n_observations, np.nan),
        )
        t_features = t_classified = perf_counter()
    else:
        try:
            features = extract_candidate_features(track, nodes, arcs, candidates, trip_id)
        except (ValueError, OSError) as error:
            logger.warning("Feature extraction failed for %s: %s", trip_id, error)
            raise FeatureExtractionError(
                f"Features could not be computed for trip '{trip_id}'."
            ) from error
        t_features = perf_counter()

        # Only the agreed feature columns reach the classifier.
        probabilities = classifier.predict_proba(features.loc[:, list(FEATURE_COLUMNS)])
        t_classified = perf_counter()

        feature_rows = features.assign(_candidate_probability=probabilities)
        frame = feature_rows.merge(
            candidates[
                ["gps_observation_index", "candidate_arc_id", "candidate_rank", "distance_m"]
            ],
            on=["gps_observation_index", "candidate_arc_id"],
            how="left",
            validate="one_to_one",
        )
        aligned_probabilities = frame.pop("_candidate_probability").to_numpy(
            dtype=np.float64
        )
        try:
            decode_result = decoder.decode(
                DecodeInput(
                    candidates=frame,
                    probabilities=aligned_probabilities,
                    n_observations=n_observations,
                    track=track,
                    nodes=nodes,
                    arcs=arcs,
                )
            )
        except ApiError:
            raise
        except Exception as error:
            logger.exception("Decoder '%s' failed for trip %s", options.decoder, trip_id)
            raise DecoderError("The decoder failed for this trip.") from error
    arc_ids, scores = _validated_decode_result(decode_result, n_observations, len(arcs))
    t_decoded = perf_counter()

    matched = arc_ids >= 0
    n_matched = int(matched.sum())
    sequence = _collapse_consecutive(arc_ids[matched])
    geometry = build_geometry(sequence, nodes, arcs)

    candidate_counts = generated.observations["candidate_count"].to_numpy(dtype=np.int64)
    matches = [
        ObservationMatch(
            gps_observation_index=index,
            candidate_count=int(candidate_counts[index]),
            arc_id=int(arc_ids[index]) if matched[index] else None,
            probability=(
                float(scores[index])
                if matched[index] and np.isfinite(scores[index])
                else None
            ),
        )
        for index in range(n_observations)
    ]
    matched_scores = scores[matched & np.isfinite(scores)]

    if candidates.empty:
        status = "no_candidates"
    elif n_matched == 0:
        status = "no_match"
    elif n_matched < n_observations:
        status = "partial"
    else:
        status = "ok"

    def ms(start: float, end: float) -> float:
        return round((end - start) * 1000.0, 3)

    finished = perf_counter()
    return PredictResponse(
        trip_id=trip_id,
        status=status,
        model=classifier.name,
        decoder=options.decoder,
        gps_points=_valid_gps_points(track),
        predicted_arc_ids=[int(arc_id) for arc_id in sequence],
        geometry=geometry,
        matches=matches,
        metrics=PredictMetrics(
            n_gps_observations=n_observations,
            n_observations_with_candidates=int((candidate_counts > 0).sum()),
            n_candidates=int(len(candidates)),
            n_matched_observations=n_matched,
            match_coverage=(n_matched / n_observations) if n_observations else 0.0,
            n_predicted_arcs=int(len(sequence)),
            mean_match_probability=(
                float(matched_scores.mean()) if matched_scores.size else None
            ),
            min_match_probability=(
                float(matched_scores.min()) if matched_scores.size else None
            ),
            search_radius_m=options.search_radius_m,
            max_candidates=options.max_candidates,
            timings_ms=TimingsMs(
                load=ms(started, t_loaded),
                candidates=ms(t_loaded, t_candidates),
                features=ms(t_candidates, t_features),
                classify=ms(t_features, t_classified),
                decode=ms(t_classified, t_decoded),
                total=ms(started, finished),
            ),
        ),
    )
