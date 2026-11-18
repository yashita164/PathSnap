"""Adapter from the API decoder contract to the shared HMM/Viterbi decoder."""

from __future__ import annotations

import numpy as np

from api.decoding import DecodeInput, DecodeResult, register_decoder
from src.hmm import decode_route


class HmmViterbiDecoder:
    name = "hmm"

    def decode(self, data: DecodeInput) -> DecodeResult:
        result = decode_route(
            data.track,
            data.nodes,
            data.arcs,
            data.candidates,
            candidate_probabilities=data.probabilities,
        )
        return DecodeResult(
            arc_ids=result.predictions["hmm_arc_id"].to_numpy(dtype=np.int64),
            scores=result.predictions["hmm_candidate_probability"].to_numpy(
                dtype=np.float64
            ),
        )


def register_hmm_decoder() -> None:
    register_decoder("hmm", HmmViterbiDecoder, overwrite=True)
