#!/usr/bin/env python3
"""Run all 20 ordered cross-species threshold transfers.

Threshold selection uses PPI-Caliper and the shared within-species protocol.
All 5 x 4 ordered cross-species directions are evaluated.
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
from transfer_species.plot import save_transfer_figures

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.PPICaliper import PPICaliper
from utils.sampling import stratified_half_split
from utils.evaluation import evaluate_threshold, mean_confidence_interval
from utils.inputs import (
    CHECKPOINTS,
    SPECIES,
    load_inputs,
    parse_experiment_args,
    sha256_indices,
)


SPECIES_PAIRS = tuple(permutations(SPECIES, 2))


def run_one_model_scenario_seed(
    source, target, source_split, target_split, q_levels, *,
    settings, experiment, split_sizes, split_hashes,
) -> list[dict[str, object]]:
    """Apply the source threshold to the source holdout and the target test batch."""
    calibration_idx, holdout_idx = source_split
    mapping_idx, test_idx = target_split
    source = source[["score", "label"]]
    calibration = source.iloc[calibration_idx]
    holdout = source.iloc[holdout_idx]
    mapping_scores = target["score"].iloc[mapping_idx]
    test = target[["score", "label"]].iloc[test_idx]

    source_caliper = PPICaliper(**settings)
    source_caliper.calibration(calibration["score"], calibration["label"], holdout["score"])
    target_caliper = PPICaliper(**settings)
    target_caliper.calibration(
        calibration["score"], calibration["label"], test["score"], target_mapping_scores=mapping_scores,
    )

    rows: list[dict[str, object]] = []
    for q in q_levels:
        threshold = target_caliper.select_quantile_threshold(q)
        # Both batches use the same source risk grid, but have their own score mappings.
        source_cutoff = source_caliper.find_raw_score_cutoff(threshold)
        target_cutoff = target_caliper.find_raw_score_cutoff(threshold)
        source_metrics = evaluate_threshold(holdout["score"], holdout["label"], source_cutoff)
        target_metrics = evaluate_threshold(test["score"], test["label"], target_cutoff)
        rows.append(
            {
                **experiment,
                "target_fdr": q,
                "quantile_threshold": threshold,
                "source_cutoff_score": source_cutoff,
                "target_cutoff_score": target_cutoff,
                "calibration_risk": target_caliper.calibration_risk_,
                "lambda_suffix_condition_exists": bool(np.isfinite(threshold)),
                "source_calibration_fraction_requested": 0.5,
                "target_mapping_fraction_requested": 0.5,
                "target_labels_used_for_threshold": False,
                **split_sizes,
                "n_threshold_calibration_pool": split_sizes["n_source_calibration"],
                "n_pseudoqueries": target_caliper.n_pseudoqueries,
                "pseudoquery_size": target_caliper.query_size,
                "quantile_anchors_requested": target_caliper.n_anchors,
                "lambda_grid_size": target_caliper.n_lambdas,
                "delta": target_caliper.delta,
                **{f"source_holdout_{name}": value for name, value in source_metrics.items()},
                **{f"target_{name}": value for name, value in target_metrics.items()},
                "target_controlled": target_metrics["observed_FDP"] <= q,
                **split_hashes,
            }
        )
    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = results.groupby(["model", "scenario_id", "target_fdr"], sort=False)
    for (model, scenario_id, q), sub in grouped:
        target_fdp = sub["target_observed_FDP"].to_numpy(dtype=np.float64)
        mean, sd, ci_low, ci_high = mean_confidence_interval(target_fdp)
        first = sub.iloc[0]
        rows.append(
            {
                "model": model,
                "scenario_id": scenario_id,
                "scenario_family": first["scenario_family"],
                "source_species": first["source_species"],
                "target_species": first["target_species"],
                "target_fdr": float(q),
                "n_seeds": int(sub["seed"].nunique()),
                "target_observed_FDP_mean": mean,
                "target_observed_FDP_sd": sd,
                "target_observed_FDP_ci95_low": ci_low,
                "target_observed_FDP_ci95_high": ci_high,
                "target_mean_minus_q": mean - float(q),
                "target_anti_conservative_mean": bool(mean > float(q)),
                "target_controlled_seed_fraction": float(sub["target_controlled"].mean()),
                "target_empty_discovery_seed_fraction": float(
                    sub["target_empty_discovery_set"].mean()
                ),
                "target_n_selected_mean": float(sub["target_n_selected"].mean()),
                "target_recall_mean": float(sub["target_recall"].mean()),
                "source_holdout_observed_FDP_mean": float(
                    sub["source_holdout_observed_FDP"].mean()
                ),
                "quantile_threshold_mean": float(sub["quantile_threshold"].mean()),
                "quantile_threshold_sd": (
                    float(sub["quantile_threshold"].std(ddof=1))
                    if np.isfinite(sub["quantile_threshold"]).all()
                    else float("nan")
                ),
                "calibration_risk_mean": float(sub["calibration_risk"].mean()),
                "lambda_suffix_failure_count": int((~sub["lambda_suffix_condition_exists"]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "scenario_id", "target_fdr"], ignore_index=True)


def compute_classification_metrics(
    score_tables: dict[tuple[str, str], pd.DataFrame], input_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Compute full-dataset AUPRC and AUROC from the released scores."""

    metadata = ["model", "checkpoint_repository", "species", "n_rows", "n_positive",
                "n_negative", "positive_fraction"]
    metrics = input_manifest[metadata].copy()
    auprc, auroc = [], []
    for row in metrics.itertuples(index=False):
        table = score_tables[row.model, row.species]
        auprc.append(average_precision_score(table["label"], table["score"]))
        auroc.append(roc_auc_score(table["label"], table["score"]))
    return metrics.assign(average_precision_aupr=auprc, roc_auc=auroc)


def main() -> None:
    args = parse_experiment_args(__doc__, "transfer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"

    if args.plots_only:
        results_path = args.output_dir / "all_seed_results.csv"
        metrics_path = args.output_dir / "classification_metrics.csv"
        results = pd.read_csv(results_path)
        metrics = pd.read_csv(metrics_path)
        save_transfer_figures(results, metrics, args.predictor, figure_dir)
        print(f"Redrew transfer figures from {args.output_dir}", flush=True)
        return

    checkpoints = CHECKPOINTS[args.predictor]
    q_levels = sorted(args.q_levels)
    score_tables, input_manifest = load_inputs(args.predictor, args.data_root)
    input_manifest.to_csv(args.output_dir / "input_manifest.csv", index=False)

    all_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    for source_species, target_species in SPECIES_PAIRS:
        source_labels = score_tables["humanv11", source_species]["label"].to_numpy()
        target_labels = score_tables["humanv11", target_species]["label"].to_numpy()
        source_index = SPECIES.index(source_species)
        target_index = SPECIES.index(target_species)
        scenario_id = f"{source_species}_to_{target_species}"
        query_size = round(len(source_labels) / 10)
        for seed in seeds:
            calibration_idx, holdout_idx = stratified_half_split(source_labels, seed, source_index)
            mapping_idx, test_idx = stratified_half_split(target_labels, seed, target_index)
            split_sizes = {
                "n_source_full": len(source_labels),
                "n_source_calibration": len(calibration_idx),
                "n_source_holdout": len(holdout_idx),
                "n_target_full": len(target_labels),
                "n_target_mapping": len(mapping_idx),
                "n_target_test": len(test_idx),
            }
            split_hashes = {
                "source_calibration_sha256": sha256_indices(calibration_idx),
                "source_holdout_sha256": sha256_indices(holdout_idx),
                "target_mapping_sha256": sha256_indices(mapping_idx),
                "target_test_sha256": sha256_indices(test_idx),
            }
            split_rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_family": "pure_source",
                    "seed": seed,
                    "source_species": source_species,
                    "target_species": target_species,
                    **split_sizes,
                    "n_pseudoqueries": 10,
                    "pseudoquery_size": query_size,
                    **split_hashes,
                }
            )
            settings = {
                "seed": seed,
                "species_index": source_index,
                "query_size": query_size,
                "n_anchors": args.quantile_anchors,
                "n_lambdas": args.lambda_grid_size,
                "delta": args.delta,
            }
            for model, repository in checkpoints.items():
                experiment = {
                    "model": model,
                    "checkpoint_repository": repository,
                    "scenario_id": scenario_id,
                    "scenario_family": "pure_source",
                    "source_species": source_species,
                    "target_species": target_species,
                    "seed": seed,
                }
                all_rows.extend(
                    run_one_model_scenario_seed(
                        score_tables[model, source_species], score_tables[model, target_species],
                        (calibration_idx, holdout_idx), (mapping_idx, test_idx), q_levels,
                        settings=settings, experiment=experiment,
                        split_sizes=split_sizes, split_hashes=split_hashes,
                    )
                )
        print(
            f"Completed {source_species}_to_{target_species}: {args.n_seeds} paired seeds × 2 checkpoints",
            flush=True,
        )

    results = (
        pd.DataFrame(all_rows)
        .sort_values(["model", "scenario_id", "seed", "target_fdr"])
        .reset_index(drop=True)
    )
    summary = summarize_results(results)
    split_manifest = pd.DataFrame(split_rows).sort_values(["scenario_id", "seed"])
    classification_metrics = compute_classification_metrics(score_tables, input_manifest)
    results.to_csv(args.output_dir / "all_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    split_manifest.to_csv(args.output_dir / "split_manifest.csv", index=False)
    classification_metrics.to_csv(args.output_dir / "classification_metrics.csv", index=False)

    if not args.no_plots:
        save_transfer_figures(results, classification_metrics, args.predictor, figure_dir)
    protocol = {
        "experiment": "Cross-species calibration",
        "predictor": args.predictor,
        "models": checkpoints,
        "scenarios": [f"{source}_to_{target}" for source, target in SPECIES_PAIRS],
        "seeds": list(seeds),
        "target_fdr": list(q_levels),
        "source_split": "50% calibration and 50% holdout, stratified by label",
        "target_split": "50% mapping and 50% test, stratified by label",
        "threshold_labels": "source calibration labels",
        "quantile_anchors": args.quantile_anchors,
        "mapping_samples": "source calibration scores and target mapping scores",
        "n_pseudoqueries": 10,
        "pseudoquery_size": "round(full source species N / 10)",
        "pseudoquery_sampling": "uniform with replacement from source calibration half",
        "lambda_grid_size": args.lambda_grid_size,
        "delta": args.delta,
        "threshold_rule": "first threshold whose entire suffix passes",
        "no_passing_threshold": "select no pairs",
        "empty_discovery_FDP": 0.0,
        "uncertainty": "95% Student-t confidence interval for mean FDP",
        "classification_metrics": "AUPR and AUROC on complete score tables",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote threshold-transfer outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
