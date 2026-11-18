"""Nested, trip-grouped Optuna search for Logistic Regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from src.model_evaluation import (
    _classification_metrics,
    _top1_metrics,
    load_training_data,
    prepare_evaluation_data,
)
from src.models.logistic_regression import (
    predict_model,
    predict_proba_model,
    train_model,
)


OUTER_SPLITS = 5
INNER_SPLITS = 4
MIN_C = 1e-4
MAX_C = 1e3
MAX_ITER = 1000
SOLVER = "lbfgs"
IMPUTE_STRATEGY = "median"


def _group_kfold_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic GroupKFold splits and reject any trip overlap."""
    if groups.nunique() < n_splits:
        raise ValueError(
            f"GroupKFold with {n_splits} folds requires at least {n_splits} trips."
        )
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_groups = set(groups.iloc[train_indices])
        validation_groups = set(groups.iloc[validation_indices])
        if train_groups.intersection(validation_groups):
            raise RuntimeError(f"Trip leakage detected in fold {fold_number}.")
    return splits


def _score_c_on_folds(
    C: float,
    X: pd.DataFrame,
    y: np.ndarray,
    metadata: pd.DataFrame,
    labelled_observations: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, list[float], list[float]]:
    top1_scores: list[float] = []
    log_losses: list[float] = []
    for train_indices, validation_indices in splits:
        X_train = X.iloc[train_indices]
        y_train = y[train_indices]
        X_validation = X.iloc[validation_indices]
        y_validation = y[validation_indices]
        validation_metadata = metadata.iloc[validation_indices].reset_index(drop=True)
        validation_trips = set(validation_metadata["trip_id"])

        model = train_model(
            X_train,
            y_train,
            C=C,
            max_iter=MAX_ITER,
            solver=SOLVER,
            random_state=42,
            impute_strategy=IMPUTE_STRATEGY,
        )
        probabilities = predict_proba_model(model, X_validation)
        top1, evaluated, _, _, _ = _top1_metrics(
            validation_metadata,
            labelled_observations,
            probabilities,
            validation_trips,
            y_validation,
        )
        if evaluated == 0 or not np.isfinite(top1):
            raise ValueError("An inner validation fold has no scorable Top-1 observations.")
        top1_scores.append(top1)
        log_losses.append(
            float(
                log_loss(
                    y_validation,
                    np.column_stack((1.0 - probabilities, probabilities)),
                    labels=[0, 1],
                )
            )
        )
    return float(np.mean(top1_scores)), top1_scores, log_losses


def _trial_rows(
    study: optuna.Study,
    outer_fold: int,
    sampler_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        duration = (
            trial.duration.total_seconds()
            if trial.duration is not None
            else float("nan")
        )
        rows.append(
            {
                "outer_fold": outer_fold,
                "trial_number": trial.number,
                "C": trial.params.get("C", np.nan),
                "objective_value": trial.value,
                "status": trial.state.name,
                "duration_seconds": duration,
                "sampler_seed": sampler_seed,
                "inner_fold_top1_scores": json.dumps(
                    trial.user_attrs.get("inner_fold_top1_scores", [])
                ),
                "inner_fold_log_losses": json.dumps(
                    trial.user_attrs.get("inner_fold_log_losses", [])
                ),
            }
        )
    return rows


def _save_trial_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    columns = [
        "outer_fold",
        "trial_number",
        "C",
        "objective_value",
        "status",
        "duration_seconds",
        "sampler_seed",
        "inner_fold_top1_scores",
        "inner_fold_log_losses",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def _save_optimization_history(
    studies: list[optuna.Study],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for outer_fold, study in enumerate(studies, start=1):
        completed = [
            trial for trial in study.trials if trial.value is not None
        ]
        if not completed:
            continue
        trial_numbers = [trial.number + 1 for trial in completed]
        values = [float(trial.value) for trial in completed]
        best_so_far = np.maximum.accumulate(values)
        ax.plot(
            trial_numbers,
            values,
            marker="o",
            linewidth=1,
            alpha=0.55,
            label=f"Outer fold {outer_fold} trials",
        )
        ax.plot(
            trial_numbers,
            best_so_far,
            linestyle="--",
            linewidth=1.5,
            label=f"Outer fold {outer_fold} best so far",
        )
    ax.set_title("Optuna Logistic Regression Top-1 Search History")
    ax.set_xlabel("Trial number within outer-fold study")
    ax.set_ylabel("Mean inner-fold observation-level Top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(
    output_path: Path,
    trials: pd.DataFrame,
    outer_results: pd.DataFrame,
    best_c: float,
    fold_best_cs: list[float],
    n_trials: int,
    seed: int,
    baseline_comparison: pd.DataFrame | None,
) -> None:
    outer_metrics = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "top1_accuracy",
        "training_seconds",
        "inference_seconds",
    )
    lines = [
        "# PathSnap Phase 6 Logistic Regression Optimization",
        "",
        "## Search configuration",
        "",
        f"- Total Optuna trials: {n_trials}, allocated across five outer folds.",
        "- Each outer training partition has its own study with a four-fold "
        "`GroupKFold` objective; six trials are used per fold for the initial "
        "30-trial run.",
        "- Sampler: seeded `TPESampler`; sampler seed is `base_seed + outer_fold - 1`. "
        "Each study uses two startup trials and runs sequentially.",
        f"- Objective: maximize mean inner-fold observation-level Top-1 accuracy. "
        f"`C` is searched log-uniformly in [{MIN_C:g}, {MAX_C:g}].",
        f"- Fixed classifier settings: solver `{SOLVER}`, max_iter `{MAX_ITER}`, "
        "random_state `42`; median imputation and scaling remain inside the "
        "Logistic Regression pipeline.",
        f"- Outer-fold best C values: `{[float(value) for value in fold_best_cs]}`.",
        f"- Final reported C: **{best_c:.8g}**, the median of the five outer-fold "
        "best C values. No additional study was run beyond the 30-trial budget.",
        "",
        "## Nested outer-fold results",
        "",
        "Mean ± standard deviation across the five held-out-trip outer folds:",
        "",
        "| Metric | Mean ± standard deviation |",
        "|---|---:|",
    ]
    for metric in outer_metrics:
        values = pd.to_numeric(outer_results[metric], errors="coerce").dropna()
        mean = float(values.mean()) if not values.empty else float("nan")
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        lines.append(f"| {metric} | {mean:.4f} ± {std:.4f} |")
    lines.extend(
        [
            "",
            "All five outer folds had zero trip overlap between training and "
            "validation. Inner-fold preprocessing was fitted only on each inner "
            "training partition; outer evaluation preprocessing was fitted only on "
            "the corresponding outer training partition.",
            "",
            "## Phase 5 baseline comparison",
            "",
        ]
    )
    if baseline_comparison is not None:
        baseline_rows = baseline_comparison.loc[
            baseline_comparison["model"].eq("Logistic Regression")
        ]
        if baseline_rows.empty:
            raise ValueError(
                "The saved Phase 5 comparison has no Logistic Regression row."
            )
        baseline = baseline_rows.iloc[0]
        lines.extend(
            [
                f"- Phase 5 five-fold GroupKFold Top-1 accuracy: "
                f"{baseline['top1_accuracy_mean']:.4f} ± "
                f"{baseline['top1_accuracy_std']:.4f}.",
                f"- Phase 5 log loss: {baseline['log_loss_mean']:.4f} ± "
                f"{baseline['log_loss_std']:.4f}.",
                "- The Phase 5 comparison file was read only and was not overwritten. "
                "Its score is a non-nested baseline; the optimized score above is "
                "from nested outer folds, so the protocol difference should be "
                "considered when comparing them.",
            ]
        )
    else:
        lines.append("- No Phase 5 comparison table was supplied.")
    lines.extend(
        [
            "",
            "## Trial accounting and limitations",
            "",
            f"- Trial rows recorded: {len(trials)}; statuses: "
            f"`{trials['status'].value_counts(dropna=False).to_dict()}`.",
            "- Only the prepared training candidate and observation datasets were "
            "used. Ambiguous/unreliable labels are excluded by the shared Phase 5 "
            "preparation helper; trip and reference-label columns are metadata, "
            "not model features.",
            "- The final C is an aggregate of fold-specific optima. It is not the "
            "result of a separate full-data Optuna search because the trial budget "
            "was capped at 30.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def optimize_logistic_regression(
    candidate_features: pd.DataFrame,
    observations: pd.DataFrame,
    output_directory: str | Path,
    figures_directory: str | Path,
    *,
    n_trials: int = 30,
    seed: int = 42,
    baseline_comparison: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the 30-total-trial nested GroupKFold Optuna search."""
    if n_trials < OUTER_SPLITS:
        raise ValueError(
            f"n_trials must be at least {OUTER_SPLITS}, one trial per outer fold."
        )
    prepared = prepare_evaluation_data(candidate_features, observations)
    outer_splits = _group_kfold_splits(
        prepared.X,
        prepared.y,
        prepared.metadata["trip_id"],
        OUTER_SPLITS,
    )
    output_path = Path(output_directory)
    figures_path = Path(figures_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    trial_csv = output_path / "optuna_trials.csv"
    outer_csv = output_path / "nested_cv_results.csv"
    best_json = output_path / "best_parameters.json"
    summary_md = output_path / "optimization_summary.md"
    history_png = figures_path / "phase6_optimization_history.png"

    trials_per_fold, extra_trials = divmod(n_trials, OUTER_SPLITS)
    trial_allocation = [
        trials_per_fold + int(fold_index < extra_trials)
        for fold_index in range(OUTER_SPLITS)
    ]
    all_trial_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    studies: list[optuna.Study] = []
    fold_best_cs: list[float] = []

    for outer_index, (outer_train_indices, outer_validation_indices) in enumerate(
        outer_splits
    ):
        outer_fold = outer_index + 1
        outer_train_metadata = prepared.metadata.iloc[outer_train_indices].reset_index(
            drop=True
        )
        outer_validation_metadata = prepared.metadata.iloc[
            outer_validation_indices
        ].reset_index(drop=True)
        outer_train_trips = set(outer_train_metadata["trip_id"])
        outer_validation_trips = set(outer_validation_metadata["trip_id"])
        if outer_train_trips.intersection(outer_validation_trips):
            raise RuntimeError(f"Trip leakage detected in outer fold {outer_fold}.")

        X_outer_train = prepared.X.iloc[outer_train_indices].reset_index(drop=True)
        y_outer_train = prepared.y[outer_train_indices]
        inner_splits = _group_kfold_splits(
            X_outer_train,
            y_outer_train,
            outer_train_metadata["trip_id"],
            INNER_SPLITS,
        )

        sampler_seed = seed + outer_index
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=sampler_seed, n_startup_trials=2),
            study_name=f"logistic_regression_outer_fold_{outer_fold}",
        )
        studies.append(study)

        def objective(trial: optuna.Trial) -> float:
            C = trial.suggest_float("C", MIN_C, MAX_C, log=True)
            objective_value, top1_scores, log_losses = _score_c_on_folds(
                C,
                X_outer_train,
                y_outer_train,
                outer_train_metadata,
                prepared.labelled_observations,
                inner_splits,
            )
            trial.set_user_attr("inner_fold_top1_scores", top1_scores)
            trial.set_user_attr("inner_fold_log_losses", log_losses)
            return objective_value

        try:
            study.optimize(
                objective,
                n_trials=trial_allocation[outer_index],
                n_jobs=1,
                show_progress_bar=False,
            )
        finally:
            all_trial_rows.extend(
                _trial_rows(study, outer_fold, sampler_seed)
            )
            _save_trial_rows(all_trial_rows, trial_csv)

        if not study.trials or study.best_trial.value is None:
            raise RuntimeError(f"Outer fold {outer_fold} has no completed Optuna trial.")
        best_c = float(study.best_params["C"])
        fold_best_cs.append(best_c)

        X_outer_validation = prepared.X.iloc[outer_validation_indices]
        y_outer_validation = prepared.y[outer_validation_indices]
        start = perf_counter()
        outer_model = train_model(
            prepared.X.iloc[outer_train_indices],
            prepared.y[outer_train_indices],
            C=best_c,
            max_iter=MAX_ITER,
            solver=SOLVER,
            random_state=42,
            impute_strategy=IMPUTE_STRATEGY,
        )
        training_seconds = perf_counter() - start
        start = perf_counter()
        predictions = predict_model(outer_model, X_outer_validation)
        probabilities = predict_proba_model(outer_model, X_outer_validation)
        inference_seconds = perf_counter() - start
        metrics = _classification_metrics(
            y_outer_validation, predictions, probabilities
        )
        (
            top1_accuracy,
            top1_observations,
            top1_unavailable,
            multiple_positive_groups,
            tie_groups,
        ) = _top1_metrics(
            outer_validation_metadata,
            prepared.labelled_observations,
            probabilities,
            outer_validation_trips,
            y_outer_validation,
        )
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "best_C": best_c,
                "inner_best_top1_accuracy": float(study.best_value),
                **metrics,
                "top1_accuracy": top1_accuracy,
                "top1_observations_evaluated": top1_observations,
                "top1_observations_without_available_correct_candidate": (
                    top1_unavailable
                ),
                "top1_multiple_positive_groups": multiple_positive_groups,
                "top1_probability_tie_groups": tie_groups,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "n_train_samples": int(len(outer_train_indices)),
                "n_validation_samples": int(len(outer_validation_indices)),
                "n_train_trips": len(outer_train_trips),
                "n_validation_trips": len(outer_validation_trips),
                "trip_overlap_count": 0,
                "validation_trip_ids": "|".join(sorted(outer_validation_trips)),
                "inner_trial_count": trial_allocation[outer_index],
            }
        )
        pd.DataFrame(outer_rows).to_csv(outer_csv, index=False)

    final_c = float(np.median(fold_best_cs))
    trial_frame = pd.DataFrame(all_trial_rows)
    outer_frame = pd.DataFrame(outer_rows)
    _save_trial_rows(all_trial_rows, trial_csv)
    _save_optimization_history(studies, history_png)

    best_parameters = {
        "model": "Logistic Regression",
        "parameters": {
            "C": final_c,
            "max_iter": MAX_ITER,
            "solver": SOLVER,
            "random_state": 42,
            "impute_strategy": IMPUTE_STRATEGY,
        },
        "selection_rule": "Median C of the five nested outer-fold Optuna best trials.",
        "outer_fold_best_C": {
            str(index + 1): value for index, value in enumerate(fold_best_cs)
        },
        "objective": "Mean observation-level Top-1 accuracy across inner folds.",
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "total_trials": n_trials,
        "trial_allocation_by_outer_fold": {
            str(index + 1): value for index, value in enumerate(trial_allocation)
        },
        "sampler": {
            "name": "TPESampler",
            "base_seed": seed,
            "n_startup_trials": 2,
            "n_jobs": 1,
        },
    }
    best_json.write_text(
        json.dumps(best_parameters, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary(
        summary_md,
        trial_frame,
        outer_frame,
        final_c,
        fold_best_cs,
        n_trials,
        seed,
        baseline_comparison,
    )
    return {
        "trials": trial_frame,
        "outer_results": outer_frame,
        "best_parameters": best_parameters,
        "best_C": final_c,
        "output_directory": output_path,
    }


def run_phase6_optimization(
    project_root: str | Path = ".",
    *,
    n_trials: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Run optimization using only the prepared training artifacts."""
    root = Path(project_root)
    candidates, observations = load_training_data(
        root / "data" / "processed" / "train_candidate_features.csv",
        root / "data" / "processed" / "train_observations.csv",
    )
    baseline_path = root / "reports" / "model_comparison.csv"
    baseline_comparison = pd.read_csv(baseline_path)
    return optimize_logistic_regression(
        candidates,
        observations,
        root / "reports" / "optimization",
        root / "reports" / "figures",
        n_trials=n_trials,
        seed=seed,
        baseline_comparison=baseline_comparison,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_phase6_optimization(
        Path(__file__).resolve().parents[2],
        n_trials=args.trials,
        seed=args.seed,
    )
    print(f"Completed {len(result['trials'])} trials.")
    print(f"Selected Logistic Regression C={result['best_C']:.8g}")
    print(result["outer_results"].to_string(index=False))


if __name__ == "__main__":
    main()
