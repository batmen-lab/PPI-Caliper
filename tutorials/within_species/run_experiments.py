#!/usr/bin/env python3
"""Run within-species PPI-Caliper evaluation on fixed PLM-Interact or MINT scores.

Uses released prediction scores in ``data/scores``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.PPICaliper import PPICaliper
from utils.sampling import stratified_half_split
from utils.evaluation import evaluate_threshold, mean_confidence_interval
from within_species.plot import save_within_figures
from utils.inputs import (
    CHECKPOINTS,
    SPECIES,
    load_inputs,
    parse_experiment_args,
    sha256_indices,
)


def run_one_model_seed(
    *,
    df_scores: pd.DataFrame,
    input_row,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_counts: dict[str, int],
    split_hashes: dict[str, str],
    q_levels: tuple[float, ...],
    quantile_anchors: int,
    lambda_grid_size: int,
    delta: float,
) -> list[dict[str, object]]:
    scores = df_scores["score"].to_numpy(dtype=np.float64)
    labels = df_scores["label"].to_numpy(dtype=np.int8)
    train_scores = scores[train_idx]
    train_labels = labels[train_idx]
    test_scores = scores[test_idx]
    test_labels = labels[test_idx]

    caliper = PPICaliper(
        seed=seed,
        species_index=SPECIES.index(input_row.species),
        n_pseudoqueries=split_counts["n_pseudoqueries"],
        query_size=split_counts["pseudoquery_size"],
        n_anchors=quantile_anchors,
        n_lambdas=lambda_grid_size,
        delta=delta,
    )
    caliper.calibration(train_scores, train_labels, test_scores)
    rows: list[dict[str, object]] = []
    for q in q_levels:
        threshold = caliper.select_quantile_threshold(q)
        cutoff = caliper.find_raw_score_cutoff(threshold)
        metrics = evaluate_threshold(test_scores, test_labels, cutoff)
        rows.append(
            {
                "model": input_row.model,
                "checkpoint_repository": input_row.checkpoint_repository,
                "species": input_row.species,
                "seed": seed,
                "target_fdr": q,
                "quantile_threshold": threshold,
                "cutoff_score": cutoff,
                "calibration_risk": caliper.calibration_risk_,
                "lambda_suffix_condition_exists": bool(np.isfinite(threshold)),
                **metrics,
                "controlled": bool(float(metrics["observed_FDP"]) <= q),
                **split_counts,
                "quantile_anchors_requested": int(quantile_anchors),
                "lambda_grid_size": int(lambda_grid_size),
                "delta": float(delta),
                **split_hashes,
                "input_path": input_row.path,
                "input_sha256": input_row.sha256,
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
    return pd.DataFrame(rows).sort_values(["species", "model", "target_fdr"], ignore_index=True)


def main() -> None:
    args = parse_experiment_args(__doc__, "within")
    q_levels = tuple(sorted(args.q_levels))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"

    if args.plots_only:
        results_path = args.output_dir / "method1_all_seed_results.csv"
        results = pd.read_csv(results_path)
        save_within_figures(results, args.predictor, figure_dir)
        print(f"Redrew within-species figures from {results_path}", flush=True)
        return

    checkpoints = CHECKPOINTS[args.predictor]
    score_tables, input_manifest = load_inputs(args.predictor, args.data_root)
    input_manifest.to_csv(args.output_dir / "input_manifest.csv", index=False)
    input_rows = {
        (row.model, row.species): row for row in input_manifest.itertuples(index=False)
    }

    all_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    seeds = tuple(range(args.seed_start, args.seed_start + args.n_seeds))
    for species_index, species in enumerate(SPECIES):
        labels = score_tables[("humanv11", species)]["label"].to_numpy(dtype=np.int8)
        for seed in seeds:
            train_idx, test_idx = stratified_half_split(labels, seed, species_index)
            split_counts = {
                "n_full": len(labels),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_positive": int((labels[train_idx] == 1).sum()),
                "n_test_positive": int((labels[test_idx] == 1).sum()),
                "n_pseudoqueries": 10,
                "pseudoquery_size": round(len(labels) / 10),
            }
            split_hashes = {
                "split_train_sha256": sha256_indices(train_idx),
                "split_test_sha256": sha256_indices(test_idx),
            }
            split_rows.append({"species": species, "seed": seed, **split_counts, **split_hashes})
            for model in checkpoints:
                all_rows.extend(
                    run_one_model_seed(
                        df_scores=score_tables[(model, species)],
                        input_row=input_rows[(model, species)],
                        seed=seed,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        split_counts=split_counts,
                        split_hashes=split_hashes,
                        q_levels=q_levels,
                        quantile_anchors=args.quantile_anchors,
                        lambda_grid_size=args.lambda_grid_size,
                        delta=args.delta,
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

    if not args.no_plots:
        save_within_figures(results, args.predictor, figure_dir)
    protocol = {
        "experiment": "Within-species calibration",
        "predictor": args.predictor,
        "models": checkpoints,
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
