"""Replaceable decoding stage: turns candidate probabilities into one arc per GPS point.

Integration point for the HMM/Viterbi work. To add a decoder:

    from api.decoding import register_decoder, DecodeInput, DecodeResult

    class HmmViterbiDecoder:
        name = "hmm"
        def decode(self, data: DecodeInput) -> DecodeResult: ...

    register_decoder("hmm", HmmViterbiDecoder)

It is then selectable per request ({"decoder": "hmm"}) or globally via
PATHSNAP_DECODER. ``DecodeInput`` carries the track, nodes and arcs tables so a
transition model can be built from the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from api.errors import UnknownDecoderError

UNMATCHED = -1


@dataclass(frozen=True)
class DecodeInput:
    """Everything a decoder may use. Contains no reference-route or label data.

    ``candidates`` has one row per (GPS observation, candidate arc) with the
    columns ``gps_observation_index``, ``candidate_arc_id``, ``candidate_rank``,
    ``distance_m`` plus the feature columns. ``probabilities[i]`` is the
    classifier's P(match) for ``candidates.iloc[i]``.
    """

    candidates: pd.DataFrame
    probabilities: np.ndarray
    n_observations: int
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame


@dataclass(frozen=True)
class DecodeResult:
    """One entry per GPS observation: chosen arc ID (-1 = unmatched) and its score."""

    arc_ids: np.ndarray
    scores: np.ndarray


class Decoder(Protocol):
    name: str

    def decode(self, data: DecodeInput) -> DecodeResult: ...


class ArgmaxDecoder:
    """Classifier-only decoding: the highest-probability candidate per observation.

    Ties are broken deterministically: lower candidate_rank (closer arc) first,
    then lower arc ID. Observations without candidates stay unmatched.
    """

    name = "argmax"

    def decode(self, data: DecodeInput) -> DecodeResult:
        arc_ids = np.full(data.n_observations, UNMATCHED, dtype=np.int64)
        scores = np.full(data.n_observations, np.nan, dtype=np.float64)
        frame = data.candidates
        if len(frame) == 0:
            return DecodeResult(arc_ids, scores)

        obs = frame["gps_observation_index"].to_numpy(dtype=np.int64)
        arc = frame["candidate_arc_id"].to_numpy(dtype=np.int64)
        rank = frame["candidate_rank"].to_numpy(dtype=np.int64)
        prob = np.asarray(data.probabilities, dtype=np.float64)

        # lexsort: last key is primary -> obs, then -prob, then rank, then arc.
        order = np.lexsort((arc, rank, -prob, obs))
        first_obs, first_pos = np.unique(obs[order], return_index=True)
        chosen = order[first_pos]
        arc_ids[first_obs] = arc[chosen]
        scores[first_obs] = prob[chosen]
        return DecodeResult(arc_ids, scores)


_DECODERS: dict[str, Callable[[], Decoder]] = {"argmax": ArgmaxDecoder}


def register_decoder(
    name: str, factory: Callable[[], Decoder], *, overwrite: bool = False
) -> None:
    if name in _DECODERS and not overwrite:
        raise ValueError(f"Decoder '{name}' is already registered.")
    _DECODERS[name] = factory


def available_decoders() -> list[str]:
    return sorted(_DECODERS)


def get_decoder(name: str) -> Decoder:
    try:
        factory = _DECODERS[name]
    except KeyError:
        raise UnknownDecoderError(
            f"Unknown decoder '{name}'. Available: {available_decoders()}."
        ) from None
    return factory()
