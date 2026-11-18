"""Benchmark synthetic classifier and HMM inference without project data."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_COLUMNS
from src.hmm import decode_route
from src.models.decision_tree import predict_proba_model as decision_tree_proba
from src.models.decision_tree import train_model as train_decision_tree
from src.models.knn import predict_proba_model as knn_proba
from src.models.knn import train_model as train_knn
from src.models.logistic_regression import (
    predict_proba_model as logistic_proba,
)
from src.models.logistic_regression import train_model as train_logistic
from src.models.mlp import predict_proba_model as mlp_proba
from src.models.mlp import train_model as train_mlp
from src.models.random_forest import (
    predict_proba_model as random_forest_proba,
)
from src.models.random_forest import train_model as train_random_forest


def _measure(
    name: str,
    infer: Callable[[], object],
    rows: int,
    repeats: int,
) -> tuple[str, int, float, float, float, float]:
    infer()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        infer()
        durations.append(time.perf_counter() - start)

    tracemalloc.start()
    infer()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (
        name,
        rows,
        statistics.median(durations) * 1000.0,
        min(durations) * 1000.0,
        max(durations) * 1000.0,
        peak_bytes / (1024.0 * 1024.0),
    )


def _hmm_inputs(
    observation_count: int,
    candidates_per_observation: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    segment_length_m = 10.0
    longitude_step = segment_length_m / 111_195.0
    node_count = observation_count + candidates_per_observation
    nodes = pd.DataFrame(
        {
            "longitude": np.arange(node_count, dtype=np.float64) * longitude_step,
            "latitude": np.zeros(node_count, dtype=np.float64),
        }
    )
    arcs = pd.DataFrame(
        {
            "from_node": np.arange(node_count - 1, dtype=np.int64),
            "to_node": np.arange(1, node_count, dtype=np.int64),
        }
    )
    track = pd.DataFrame(
        {
            "longitude": (
                np.arange(observation_count, dtype=np.float64) + 0.5
            )
            * longitude_step,
            "latitude": np.zeros(observation_count, dtype=np.float64),
            "timestamp": np.arange(observation_count, dtype=np.float64),
        }
    )
    offsets = np.arange(candidates_per_observation, dtype=np.int64)
    candidates = pd.DataFrame(
        {
            "gps_observation_index": np.repeat(
                np.arange(observation_count, dtype=np.int64),
                candidates_per_observation,
            ),
            "candidate_arc_id": (
                np.arange(observation_count, dtype=np.int64)[:, None] + offsets
            ).ravel(),
        }
    )
    probabilities = rng.uniform(
        0.05, 1.0, size=len(candidates)
    ).astype(np.float64)
    return track, nodes, arcs, candidates, probabilities


def run_benchmarks(
    candidate_rows: int = 5_000,
    observations: int = 100,
    candidates_per_observation: int = 5,
    repeats: int = 5,
    seed: int = 42,
) -> list[tuple[str, int, float, float, float, float]]:
    if min(candidate_rows, observations, candidates_per_observation, repeats) < 1:
        raise ValueError("Benchmark sizes and repeats must be positive integers.")

    rng = np.random.default_rng(seed)
    train_values = rng.normal(size=(2_000, len(FEATURE_COLUMNS)))
    train_X = pd.DataFrame(train_values, columns=FEATURE_COLUMNS)
    train_y = (train_values[:, 0] + train_values[:, 1] > 0.0).astype(np.int64)
    eval_X = pd.DataFrame(
        rng.normal(size=(candidate_rows, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    model_specs = [
        ("Logistic Regression", train_logistic, logistic_proba, {}),
        ("Decision Tree", train_decision_tree, decision_tree_proba, {}),
        (
            "Random Forest",
            train_random_forest,
            random_forest_proba,
            {"n_estimators": 100, "n_jobs": 1, "random_state": seed},
        ),
        (
            "MLP",
            train_mlp,
            mlp_proba,
            {
                "hidden_layer_sizes": (16,),
                "max_iter": 100,
                "early_stopping": False,
                "random_state": seed,
            },
        ),
        ("KNN", train_knn, knn_proba, {}),
    ]

    results = []
    for name, train, predict_proba, parameters in model_specs:
        model = train(train_X, train_y, **parameters)
        results.append(
            _measure(
                name,
                lambda model=model, predict_proba=predict_proba: predict_proba(
                    model, eval_X
                ),
                candidate_rows,
                repeats,
            )
        )

    hmm_rng = np.random.default_rng(seed)
    track, nodes, arcs, candidates, probabilities = _hmm_inputs(
        observations, candidates_per_observation, hmm_rng
    )
    results.append(
        _measure(
            "HMM",
            lambda: decode_route(track, nodes, arcs, candidates, probabilities),
            len(candidates),
            repeats,
        )
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rows", type=int, default=5_000)
    parser.add_argument("--observations", type=int, default=100)
    parser.add_argument("--candidates-per-observation", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_benchmarks(
        candidate_rows=args.candidate_rows,
        observations=args.observations,
        candidates_per_observation=args.candidates_per_observation,
        repeats=args.repeats,
        seed=args.seed,
    )
    print(
        "model\trows\tmedian_ms\tmin_ms\tmax_ms\ttracemalloc_peak_mib "
        "(Python-traced allocations only)"
    )
    for name, rows, median_ms, min_ms, max_ms, peak_mib in results:
        print(
            f"{name}\t{rows}\t{median_ms:.3f}\t{min_ms:.3f}\t"
            f"{max_ms:.3f}\t{peak_mib:.3f}"
        )


if __name__ == "__main__":
    main()
