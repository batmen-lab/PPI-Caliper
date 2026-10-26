#!/usr/bin/env python3
"""Run within-species PPI-Caliper evaluation on fixed PLM-Interact or MINT scores.

Uses released prediction scores in ``data/scores``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "ppi_within_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import data_path
from utils import paths
from PPICaliper import PPICaliper
from utils.sampling import stratified_half_split
from utils.evaluation import evaluate_threshold, mean_confidence_interval
from within_species.plot import plot_results
from utils.figure_style import save_figure
from utils.inputs import (
    DEFAULT_Q_LEVELS,
    MODEL_REPOSITORIES,
    SPECIES,
    configure_inputs,
    load_inputs,
    prediction_path,
    sha256_indices,
)


def run_one_model_seed(
    *,
    df_scores: pd.DataFrame,
    model: str,
    species: str,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    q_levels: tuple[float, ...],
    quantile_anchors: int,
    lambda_grid_size: int,
    delta: float,
    input_sha256: str,
) -> list[dict[str, object]]:
    scores = df_scores["score"].to_numpy(dtype=np.float64)
    labels = df_scores["label"].to_numpy(dtype=np.int8)
    train_scores = scores[train_idx]
    train_labels = labels[train_idx]
    test_scores = scores[test_idx]
    test_labels = labels[test_idx]

    caliper = PPICaliper(
        seed=seed,
        species_index=SPECIES.index(species),
        query_size=round(len(df_scores) / 10),
        n_anchors=quantile_anchors,
        n_lambdas=lambda_grid_size,
        delta=delta,
    )
    cutoffs = caliper.ppicaliper(train_scores, train_labels, test_scores, q_levels)
    rows: list[dict[str, object]] = []
    for selection in cutoffs:
        q = selection.target_fdr
        metrics = evaluate_threshold(test_scores, test_labels, selection.cutoff_score)
        rows.append(
            {
                "model": model,
                "checkpoint_repository": MODEL_REPOSITORIES[model],
                "species": species,
                "seed": seed,
                "target_fdr": q,
                "quantile_threshold": selection.quantile_threshold,
                "cutoff_score": selection.cutoff_score,
                "calibration_risk": selection.calibration_risk,
                "lambda_suffix_condition_exists": selection.suffix_condition_exists,
                **metrics,
                "controlled": bool(float(metrics["observed_FDP"]) <= q),
                "n_full": int(len(df_scores)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_positive": int((train_labels == 1).sum()),
                "n_test_positive": int((test_labels == 1).sum()),
                "n_pseudoqueries": 10,
                "pseudoquery_size": round(len(df_scores) / 10),
                "quantile_anchors_requested": int(quantile_anchors),
                "lambda_grid_size": int(lambda_grid_size),
                "delta": float(delta),
                "split_train_sha256": sha256_indices(train_idx),
                "split_test_sha256": sha256_indices(test_idx),
                "input_path": data_path(prediction_path(model, species)),
                "input_sha256": input_sha256,
            }
        )
    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = results.groupby(["model", "species", "target_fdr"], sort=False)
    for (model, species, q), sub in grouped:
        fdp = sub["observed_FDP"].to_numpy(dtype=np.float64)
        mean, sd, ci_low, ci_high = mean_confidence_interval(fdp)
        rows.append(
            {
                "model": model,
                "species": species,
                "target_fdr": float(q),
                "n_seeds": int(sub["seed"].nunique()),
                "observed_FDP_mean": mean,
                "observed_FDP_sd": sd,
                "observed_FDP_ci95_low": ci_low,
                "observed_FDP_ci95_high": ci_high,
                "mean_minus_target": mean - float(q),
                "anti_conservative_mean": bool(mean > float(q)),
                "controlled_seed_fraction": float(sub["controlled"].mean()),
                "empty_discovery_seed_fraction": float(sub["empty_discovery_set"].mean()),
                "quantile_threshold_mean": float(sub["quantile_threshold"].mean()),
                "quantile_threshold_sd": (
                    float(sub["quantile_threshold"].std(ddof=1))
                    if np.isfinite(sub["quantile_threshold"]).all()
                    else float("nan")
                ),
                "cutoff_score_mean": float(sub["cutoff_score"].mean()),
                "n_selected_mean": float(sub["n_selected"].mean()),
                "recall_mean": float(sub["recall"].mean()),
                "calibration_risk_mean": float(sub["calibration_risk"].mean()),
                "lambda_suffix_failure_count": int((~sub["lambda_suffix_condition_exists"]).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    model_order = {model: index for index, model in enumerate(MODEL_REPOSITORIES)}
    species_order = {species: index for index, species in enumerate(SPECIES)}
    summary["_model_order"] = summary["model"].map(model_order)
    summary["_species_order"] = summary["species"].map(species_order)
    summary = summary.sort_values(["_species_order", "_model_order", "target_fdr"]).drop(
        columns=["_model_order", "_species_order"]
    )
    return summary.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Default: results/<predictor>/within")
    parser.add_argument("--predictor", choices=("plm-interact", "mint"), default="plm-interact")
    parser.add_argument("--data-root", type=Path, default=paths.PROJECT_ROOT / "data")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--q-levels", nargs="+", type=float, default=list(DEFAULT_Q_LEVELS))
    parser.add_argument("--quantile-anchors", type=int, default=1000)
    parser.add_argument("--lambda-grid-size", type=int, default=100)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Redraw figures from the saved experiment results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_inputs(args.predictor, args.data_root)
    if args.output_dir is None:
        args.output_dir = paths.PROJECT_ROOT / "results" / args.predictor / "within"
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.seed_start < 0:
        raise ValueError("--seed-start cannot be negative")
    q_levels = tuple(sorted(float(q) for q in args.q_levels))
    if not q_levels or any(not 0 < q < 1 for q in q_levels):
        raise ValueError("all --q-levels must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        results_path = args.output_dir / "method1_all_seed_results.csv"
        results = pd.read_csv(results_path)
        save_figure(plot_results(results, args.predictor), f"{args.predictor}_within_metrics", figure_dir)
        print(f"Redrew within-species figures from {results_path}", flush=True)
        return

    score_tables, input_manifest = load_inputs()
    input_manifest.to_csv(args.output_dir / "input_manifest.csv", index=False)
    sha_lookup = {
        (row.model, row.species): row.sha256 for row in input_manifest.itertuples(index=False)
    }

    all_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    seeds = tuple(range(args.seed_start, args.seed_start + args.n_seeds))
    for species_index, species in enumerate(SPECIES):
        labels = score_tables[("humanv11", species)]["label"].to_numpy(dtype=np.int8)
        for seed in seeds:
            train_idx, test_idx = stratified_half_split(labels, seed, species_index)
            split_rows.append(
                {
                    "species": species,
                    "seed": seed,
                    "n_full": len(labels),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_train_positive": int((labels[train_idx] == 1).sum()),
                    "n_test_positive": int((labels[test_idx] == 1).sum()),
                    "n_pseudoqueries": 10,
                    "pseudoquery_size": round(len(labels) / 10),
                    "split_train_sha256": sha256_indices(train_idx),
                    "split_test_sha256": sha256_indices(test_idx),
                }
            )
            for model in MODEL_REPOSITORIES:
                all_rows.extend(
                    run_one_model_seed(
                        df_scores=score_tables[(model, species)],
                        model=model,
                        species=species,
                        seed=seed,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        q_levels=q_levels,
                        quantile_anchors=args.quantile_anchors,
                        lambda_grid_size=args.lambda_grid_size,
                        delta=args.delta,
                        input_sha256=sha_lookup[(model, species)],
                    )
                )
        print(f"Completed {species}: {args.n_seeds} paired seeds × 2 checkpoints", flush=True)

    results = pd.DataFrame(all_rows)
    results = results.sort_values(["species", "model", "seed", "target_fdr"]).reset_index(drop=True)
    summary = summarize_results(results)
    split_manifest = pd.DataFrame(split_rows).sort_values(["species", "seed"])
    results.to_csv(args.output_dir / "method1_all_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "method1_summary.csv", index=False)
    split_manifest.to_csv(args.output_dir / "split_and_pseudoquery_manifest.csv", index=False)

    save_figure(plot_results(results, args.predictor), f"{args.predictor}_within_metrics", figure_dir)
    protocol = {
        "experiment": "Within-species calibration",
        "predictor": args.predictor,
        "models": MODEL_REPOSITORIES,
        "species": list(SPECIES),
        "seeds": list(seeds),
        "target_fdr": list(q_levels),
        "split": "50% calibration and 50% test, stratified by label",
        "quantile_anchors": args.quantile_anchors,
        "mapping_sample": "calibration half",
        "n_pseudoqueries": 10,
        "pseudoquery_size": "round(full species N / 10)",
        "pseudoquery_sampling": "uniform with replacement from calibration half",
        "lambda_grid_size": args.lambda_grid_size,
        "delta": args.delta,
        "threshold_rule": "first threshold whose entire suffix passes",
        "no_passing_threshold": "select no pairs",
        "empty_discovery_FDP": 0.0,
        "uncertainty": "95% Student-t confidence interval for mean FDP",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote Method 1 outputs to {args.output_dir}", flush=True)
    print(f"Seed-level rows: {len(results):,}; summary rows: {len(summary):,}", flush=True)


if __name__ == "__main__":
    main()
