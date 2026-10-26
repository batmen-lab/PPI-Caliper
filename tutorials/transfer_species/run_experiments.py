#!/usr/bin/env python3
"""Run all 20 ordered cross-species threshold transfers.

Threshold selection uses PPI-Caliper and the shared within-species protocol.
All 5 x 4 ordered cross-species directions are evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import paths


_CACHE_ROOT = Path(tempfile.gettempdir()) / "ppi_transfer_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
from transfer_species.plot import plot_metrics, plot_pr_figures

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from PPICaliper import PPICaliper
from utils.sampling import stratified_half_split
from utils.evaluation import evaluate_threshold, mean_confidence_interval
from utils.inputs import (
    DEFAULT_Q_LEVELS,
    configure_inputs,
    MODEL_REPOSITORIES,
    SPECIES,
    load_inputs,
    sha256_indices,
)


SPECIES_PAIRS = tuple(permutations(SPECIES, 2))


@dataclass(frozen=True)
class TransferSplit:
    source_calibration_idx: np.ndarray
    source_holdout_idx: np.ndarray
    target_mapping_idx: np.ndarray
    target_test_idx: np.ndarray


def make_transfer_split(
    source_species: str,
    target_species: str,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    seed: int,
) -> TransferSplit:
    """Build paired model-independent indices for one scenario and seed."""

    source_calibration_idx, source_holdout_idx = stratified_half_split(
        source_labels, seed, SPECIES.index(source_species)
    )
    target_mapping_idx, target_test_idx = stratified_half_split(
        target_labels, seed, SPECIES.index(target_species)
    )

    return TransferSplit(
        source_calibration_idx=source_calibration_idx,
        source_holdout_idx=source_holdout_idx,
        target_mapping_idx=target_mapping_idx,
        target_test_idx=target_test_idx,
    )


def prefix_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def run_one_model_scenario_seed(
    *,
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    model: str,
    source_species: str,
    target_species: str,
    split: TransferSplit,
    seed: int,
    q_levels: tuple[float, ...],
    quantile_anchors: int,
    lambda_grid_size: int,
    delta: float,
) -> list[dict[str, object]]:
    """Select Method 1 thresholds and evaluate one transfer scenario."""

    source_scores = df_source["score"].to_numpy(dtype=np.float64)
    source_labels = df_source["label"].to_numpy(dtype=np.int8)
    target_scores = df_target["score"].to_numpy(dtype=np.float64)
    target_labels = df_target["label"].to_numpy(dtype=np.int8)

    source_calibration_scores = source_scores[split.source_calibration_idx]
    calibration_labels = source_labels[split.source_calibration_idx]
    source_holdout_scores = source_scores[split.source_holdout_idx]
    target_test_scores = target_scores[split.target_test_idx]
    query_size = round(len(df_source) / 10)
    caliper = PPICaliper(
        seed=seed,
        species_index=SPECIES.index(source_species),
        query_size=query_size,
        n_anchors=quantile_anchors,
        n_lambdas=lambda_grid_size,
        delta=delta,
    )
    source_cutoffs = caliper.ppicaliper(
        source_calibration_scores, calibration_labels, source_holdout_scores, q_levels,
    )
    target_cutoffs = caliper.ppicaliper(
        source_calibration_scores, calibration_labels, target_test_scores, q_levels,
        target_mapping_scores=target_scores[split.target_mapping_idx],
    )

    rows: list[dict[str, object]] = []
    for source_cutoff, selection in zip(source_cutoffs, target_cutoffs):
        q = selection.target_fdr
        source_metrics = evaluate_threshold(
            source_holdout_scores,
            source_labels[split.source_holdout_idx],
            source_cutoff.cutoff_score,
        )
        target_metrics = evaluate_threshold(
            target_test_scores,
            target_labels[split.target_test_idx],
            selection.cutoff_score,
        )
        rows.append(
            {
                "model": model,
                "checkpoint_repository": MODEL_REPOSITORIES[model],
                "scenario_id": f"{source_species}_to_{target_species}",
                "scenario_family": "pure_source",
                "source_species": source_species,
                "target_species": target_species,
                "seed": seed,
                "target_fdr": float(q),
                "quantile_threshold": float(selection.quantile_threshold),
                "source_cutoff_score": source_cutoff.cutoff_score,
                "target_cutoff_score": selection.cutoff_score,
                "calibration_risk": float(selection.calibration_risk),
                "lambda_suffix_condition_exists": bool(selection.suffix_condition_exists),
                "source_calibration_fraction_requested": 0.5,
                "target_mapping_fraction_requested": 0.5,
                "target_labels_used_for_threshold": False,
                "n_source_full": int(len(df_source)),
                "n_source_calibration": int(len(split.source_calibration_idx)),
                "n_source_holdout": int(len(split.source_holdout_idx)),
                "n_target_full": int(len(df_target)),
                "n_target_mapping": int(len(split.target_mapping_idx)),
                "n_target_test": int(len(split.target_test_idx)),
                "n_threshold_calibration_pool": int(len(calibration_labels)),
                "n_pseudoqueries": caliper.n_pseudoqueries,
                "pseudoquery_size": query_size,
                "quantile_anchors_requested": int(quantile_anchors),
                "lambda_grid_size": int(lambda_grid_size),
                "delta": float(delta),
                **prefix_metrics("source_holdout", source_metrics),
                **prefix_metrics("target", target_metrics),
                "target_controlled": bool(float(target_metrics["observed_FDP"]) <= q),
                "source_calibration_sha256": sha256_indices(split.source_calibration_idx),
                "source_holdout_sha256": sha256_indices(split.source_holdout_idx),
                "target_mapping_sha256": sha256_indices(split.target_mapping_idx),
                "target_test_sha256": sha256_indices(split.target_test_idx),
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
    summary = pd.DataFrame(rows)
    model_order = {model: index for index, model in enumerate(MODEL_REPOSITORIES)}
    scenario_order = {
        f"{source}_to_{target}": index
        for index, (source, target) in enumerate(SPECIES_PAIRS)
    }
    summary["_model_order"] = summary["model"].map(model_order)
    summary["_scenario_order"] = summary["scenario_id"].map(scenario_order)
    summary = summary.sort_values(["_model_order", "_scenario_order", "target_fdr"]).drop(
        columns=["_model_order", "_scenario_order"]
    )
    return summary.reset_index(drop=True)


def compute_classification_outputs(
    score_tables: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute full-dataset PR curves, AUPR, and AUROC from frozen scores."""

    metric_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for model in MODEL_REPOSITORIES:
        for species in SPECIES:
            df_scores = score_tables[(model, species)]
            labels = df_scores["label"].to_numpy(dtype=np.int8)
            scores = df_scores["score"].to_numpy(dtype=np.float64)
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            metric_rows.append(
                {
                    "model": model,
                    "checkpoint_repository": MODEL_REPOSITORIES[model],
                    "species": species,
                    "n_rows": int(len(df_scores)),
                    "n_positive": int((labels == 1).sum()),
                    "n_negative": int((labels == 0).sum()),
                    "positive_fraction": float((labels == 1).mean()),
                    "average_precision_aupr": float(average_precision_score(labels, scores)),
                    "roc_auc": float(roc_auc_score(labels, scores)),
                }
            )
            threshold_values = np.append(thresholds, np.nan)
            for point_index, (point_recall, point_precision, threshold) in enumerate(
                zip(recall, precision, threshold_values, strict=True)
            ):
                curve_rows.append(
                    {
                        "model": model,
                        "species": species,
                        "point_index": point_index,
                        "recall": float(point_recall),
                        "precision": float(point_precision),
                        "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    return metrics, curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: results/<predictor>/transfer",
    )
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
        args.output_dir = paths.PROJECT_ROOT / "results" / args.predictor / "transfer"
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.seed_start < 0:
        raise ValueError("--seed-start cannot be negative")
    q_levels = tuple(sorted(float(q) for q in args.q_levels))
    if not q_levels or any(not 0 < q < 1 for q in q_levels):
        raise ValueError("all q levels must be in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        results_path = args.output_dir / "all_seed_results.csv"
        metrics_path = args.output_dir / "classification_metrics.csv"
        curves_path = args.output_dir / "pr_curve_points.csv"
        results = pd.read_csv(results_path)
        metrics = pd.read_csv(metrics_path)
        curves = pd.read_csv(curves_path)
        plot_metrics(results, metrics, args.predictor, figure_dir)
        plot_pr_figures(metrics, curves, figure_dir)
        print(f"Redrew transfer figures from {args.output_dir}", flush=True)
        return

    score_tables, input_manifest = load_inputs()
    input_manifest.to_csv(args.output_dir / "input_manifest.csv", index=False)

    all_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    seeds = tuple(range(args.seed_start, args.seed_start + args.n_seeds))
    for source_species, target_species in SPECIES_PAIRS:
        source_labels = score_tables[("humanv11", source_species)]["label"].to_numpy(
            dtype=np.int8
        )
        target_labels = score_tables[("humanv11", target_species)]["label"].to_numpy(
            dtype=np.int8
        )
        for seed in seeds:
            split = make_transfer_split(
                source_species,
                target_species,
                source_labels,
                target_labels,
                seed=seed,
            )
            split_rows.append(
                {
                    "scenario_id": f"{source_species}_to_{target_species}",
                    "scenario_family": "pure_source",
                    "seed": seed,
                    "source_species": source_species,
                    "target_species": target_species,
                    "n_source_full": len(source_labels),
                    "n_source_calibration": len(split.source_calibration_idx),
                    "n_source_holdout": len(split.source_holdout_idx),
                    "n_target_full": len(target_labels),
                    "n_target_mapping": len(split.target_mapping_idx),
                    "n_target_test": len(split.target_test_idx),
                    "n_pseudoqueries": 10,
                    "pseudoquery_size": round(len(source_labels) / 10),
                    "source_calibration_sha256": sha256_indices(split.source_calibration_idx),
                    "source_holdout_sha256": sha256_indices(split.source_holdout_idx),
                    "target_mapping_sha256": sha256_indices(split.target_mapping_idx),
                    "target_test_sha256": sha256_indices(split.target_test_idx),
                }
            )
            for model in MODEL_REPOSITORIES:
                all_rows.extend(
                    run_one_model_scenario_seed(
                        df_source=score_tables[(model, source_species)],
                        df_target=score_tables[(model, target_species)],
                        model=model,
                        source_species=source_species,
                        target_species=target_species,
                        split=split,
                        seed=seed,
                        q_levels=q_levels,
                        quantile_anchors=args.quantile_anchors,
                        lambda_grid_size=args.lambda_grid_size,
                        delta=args.delta,
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
    classification_metrics, pr_curves = compute_classification_outputs(score_tables)
    results.to_csv(args.output_dir / "all_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    split_manifest.to_csv(args.output_dir / "split_manifest.csv", index=False)
    classification_metrics.to_csv(args.output_dir / "classification_metrics.csv", index=False)
    pr_curves.to_csv(args.output_dir / "pr_curve_points.csv", index=False)

    plot_metrics(results, classification_metrics, args.predictor, figure_dir)
    plot_pr_figures(classification_metrics, pr_curves, figure_dir)
    protocol = {
        "experiment": "Cross-species calibration",
        "predictor": args.predictor,
        "models": MODEL_REPOSITORIES,
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
