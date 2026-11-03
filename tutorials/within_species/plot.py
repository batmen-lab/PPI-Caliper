"""Within-species empirical FDR, power, and accuracy in Figures 2 and A1–A3."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.figure_style import MODELS, PREDICTOR_NAMES, SPECIES_NAMES, draw_bars, save_figure

METRIC_FILES = {"Empirical FDR": "fdr", "Power": "power", "Accuracy": "accuracy"}
FIGURE_NUMBERS = {
    ("plm-interact", "humanv11"): "2",
    ("plm-interact", "humanv12"): "A1",
    ("mint", "humanv11"): "A2",
    ("mint", "humanv12"): "A3",
}


def compute_within_metrics(rows):
    true_negatives = rows["n_test"] - rows["n_test_positive"] - rows["false_positives"]
    return rows.assign(
        Power=rows["true_positives"] / rows["n_test_positive"],
        Accuracy=(rows["true_positives"] + true_negatives) / rows["n_test"],
    ).rename(columns={"observed_FDP": "Empirical FDR"})


def plot_within_panel(results: pd.DataFrame, predictor: str, model: str,
                      species: str, metric: str) -> plt.Figure:
    """Draw one species and one metric from a predictor's per-split results."""
    sample = results[(results["model"] == model) & (results["species"] == species)]
    fig, ax = plt.subplots(figsize=(4.3, 4), layout="constrained")
    draw_bars(ax, compute_within_metrics(sample), metric)
    ax.set(xlabel="Target FDR", ylabel=metric,
           title=f"{PREDICTOR_NAMES[predictor]} {model} · {SPECIES_NAMES[species]}")
    return fig


def plot_within_figure(results: pd.DataFrame, predictor: str, model: str) -> plt.Figure:
    """Draw the manuscript's three metrics across the five test species."""
    checkpoint = compute_within_metrics(results[results["model"] == model])
    fig, axes = plt.subplots(3, 5, figsize=(14.5, 9), layout="constrained")
    for col, (species, label) in enumerate(SPECIES_NAMES.items()):
        sample = checkpoint[checkpoint["species"] == species]
        axes[0, col].set_title(label, weight="semibold")
        for row, metric in enumerate(METRIC_FILES):
            ax = axes[row, col]
            draw_bars(ax, sample, metric)
            ax.set_xlabel("Target FDR", fontsize=9)
            if col == 0:
                ax.set_ylabel(metric)
    figure_number = FIGURE_NUMBERS[predictor, model]
    fig.suptitle(f"Figure {figure_number} · {PREDICTOR_NAMES[predictor]} {model} · "
                 f"{checkpoint['seed'].nunique()} splits", weight="semibold")
    return fig


def save_within_figures(results: pd.DataFrame, predictor: str, output_dir: Path) -> None:
    """Export two checkpoint overviews and their individual species/metric panels."""
    output_dir = Path(output_dir)
    for model in MODELS:
        number = FIGURE_NUMBERS[predictor, model].lower()
        overview = plot_within_figure(results, predictor, model)
        save_figure(overview, f"figure_{number}_{predictor}_{model}", output_dir)
        for species in SPECIES_NAMES:
            for metric, suffix in METRIC_FILES.items():
                figure = plot_within_panel(results, predictor, model, species, metric)
                save_figure(figure, f"{species}_{suffix}", output_dir / "panels" / model)
