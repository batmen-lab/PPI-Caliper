"""Cross-species calibration and prediction figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.inputs import MODEL_REPOSITORIES, predictor_label
from utils.figure_style import (
    DISPLAY_SPECIES,
    MODELS,
    MODEL_COLORS,
    MODEL_STYLES,
    SPECIES_LABELS,
    configure_style,
    draw_mean_curve,
    save_figure,
)


def model_label(model: str) -> str:
    return "humanV11" if model == "humanv11" else "humanV12"


def plot_metrics(
    results: pd.DataFrame,
    classification_metrics: pd.DataFrame,
    predictor: str,
    figure_dir: Path,
) -> None:
    """Export FDP, power and accuracy across all target FDR levels."""
    frame = results.merge(
        classification_metrics[["model", "species", "n_positive"]].rename(
            columns={"species": "target_species"}
        ),
        on=["model", "target_species"],
    )
    target_positives = frame.n_positive - frame.n_positive // 2
    frame = frame.assign(
        accuracy=(
            frame.target_true_positives + frame.n_target_test
            - target_positives - frame.target_false_positives
        ) / frame.n_target_test,
    )
    metrics = (
        ("target_observed_FDP", "FDP"),
        ("target_recall", "Power"),
        ("accuracy", "Accuracy"),
    )
    for model in MODELS:
        for metric, label in metrics:
            figure, axes = plt.subplots(
                5, 5, figsize=(14, 14), sharex=True, sharey=True, layout="constrained"
            )
            for row, source in enumerate(DISPLAY_SPECIES):
                for column, target in enumerate(DISPLAY_SPECIES):
                    axis = axes[row, column]
                    if row == 0:
                        axis.set_title(SPECIES_LABELS[target])
                    if column == 0:
                        axis.set_ylabel(SPECIES_LABELS[source])
                    if source == target:
                        axis.set_facecolor("#EEEEEE")
                        axis.text(0.5, 0.5, "—", ha="center", va="center")
                    else:
                        selected = frame[
                            (frame.model == model)
                            & (frame.source_species == source)
                            & (frame.target_species == target)
                        ]
                        draw_mean_curve(axis, selected, metric, model)
                        if metric == "target_observed_FDP":
                            axis.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1)
                    axis.set(xlim=(0, 1), ylim=(0, 1), xticks=[0.1, 0.3, 0.5, 0.7, 0.9])
                    if row == len(DISPLAY_SPECIES) - 1:
                        axis.set_xlabel("Target FDR")
            seeds = frame.loc[frame.model == model, "seed"].nunique()
            figure.suptitle(
                f"{predictor} / {model}: cross-species {label}, {seeds} splits\n"
                "Rows: source species; columns: target species"
            )
            save_figure(figure, f"transfer_{model}_{label.lower()}", figure_dir)


def plot_heatmap(transfer_results, classification_metrics) -> plt.Figure:
    """Summarize directional FDP gaps at target FDR 0.1."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    transfer_matrices = {}
    for predictor, frame in transfer_results.items():
        for model in MODELS:
            selected = frame[(frame.model == model) & np.isclose(frame.target_fdr, 0.1)]
            matrix = (
                selected.groupby(["source_species", "target_species"])
                .target_observed_FDP.mean()
                .unstack()
            )
            transfer_matrices[(predictor, model)] = (
                matrix.reindex(index=DISPLAY_SPECIES, columns=DISPLAY_SPECIES) - 0.1
            )
    bound = max(
        float(np.nanmax(np.abs(matrix.to_numpy()))) for matrix in transfer_matrices.values()
    )
    bound = max(bound, 0.01)
    colormap = LinearSegmentedColormap.from_list("fdp_gap", ["#386A8F", "#F5F4F0", "#B36B3F"])
    colormap.set_bad("#EEEEEE")
    figure, axes = plt.subplots(2, 2, figsize=(11, 9.6), layout="constrained")
    for axis, ((predictor, model), matrix) in zip(axes.flat, transfer_matrices.items()):
        values = matrix.to_numpy()
        image = axis.imshow(
            np.ma.masked_invalid(values),
            cmap=colormap,
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
        )
        seeds = transfer_results[predictor]["seed"].nunique()
        metrics = classification_metrics[predictor]
        auprc = metrics[metrics.model == model].set_index("species").average_precision_aupr
        axis.set_xticks(
            range(5),
            [f"{SPECIES_LABELS[s]}\n{auprc[s]:.3f}" for s in DISPLAY_SPECIES],
        )
        axis.set_yticks(range(5), [SPECIES_LABELS[s] for s in DISPLAY_SPECIES])
        axis.set(
            xlabel="Target species (AUPRC)",
            ylabel="Source species",
            title=f"{predictor} / {model} ({seeds} splits)",
        )
        for row in range(5):
            for column in range(5):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    "—" if np.isnan(value) else f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white"
                    if np.isfinite(value) and abs(value) > bound * 0.65
                    else "#202020",
                )
    figure.colorbar(image, ax=list(axes.flat), shrink=0.65, label="Mean held-out FDP − 0.1")
    figure.suptitle("Directional threshold transfer at target FDR 0.1")
    return figure


def _style_pr_axis(ax: plt.Axes, *, show_xlabel: bool) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, 1.0])
    ax.tick_params(labelsize=8.0, width=0.8, length=2.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#171717")
        spine.set_linewidth(0.9)
    ax.set_ylabel("Precision", fontsize=8.5, labelpad=1.5)
    if show_xlabel:
        ax.set_xlabel("Recall", fontsize=8.5, labelpad=1.5)
    else:
        ax.tick_params(labelbottom=False)


def _draw_species_tree(ax: plt.Axes) -> None:
    """Draw a labelled relationship schematic aligned to the five PR rows."""

    y = {
        species: float(len(DISPLAY_SPECIES) - 1 - index)
        for index, species in enumerate(DISPLAY_SPECIES)
    }
    line = {"color": "#303030", "linewidth": 1.25, "solid_capstyle": "round"}
    leaf_x = 0.98
    fly_worm_x = 0.76
    animal_x = 0.56
    eukaryote_x = 0.33
    root_x = 0.10
    fly_worm_y = (y["fly"] + y["worm"]) / 2
    animal_y = (y["mouse"] + fly_worm_y) / 2
    eukaryote_y = (animal_y + y["yeast"]) / 2

    ax.plot([fly_worm_x, leaf_x], [y["fly"], y["fly"]], **line)
    ax.plot([fly_worm_x, leaf_x], [y["worm"], y["worm"]], **line)
    ax.plot([fly_worm_x, fly_worm_x], [y["worm"], y["fly"]], **line)
    ax.plot([animal_x, fly_worm_x], [fly_worm_y, fly_worm_y], **line)
    ax.plot([animal_x, leaf_x], [y["mouse"], y["mouse"]], **line)
    ax.plot([animal_x, animal_x], [fly_worm_y, y["mouse"]], **line)
    ax.plot([eukaryote_x, animal_x], [animal_y, animal_y], **line)
    ax.plot([eukaryote_x, leaf_x], [y["yeast"], y["yeast"]], **line)
    ax.plot([eukaryote_x, eukaryote_x], [y["yeast"], animal_y], **line)
    ax.plot([root_x, eukaryote_x], [eukaryote_y, eukaryote_y], **line)
    ax.plot([root_x, leaf_x], [y["ecoli"], y["ecoli"]], **line)
    ax.plot([root_x, root_x], [y["ecoli"], eukaryote_y], **line)

    for species in DISPLAY_SPECIES:
        ax.text(
            1.02,
            y[species],
            SPECIES_LABELS[species],
            ha="left",
            va="center",
            fontsize=11.0,
            fontstyle="italic" if species == "ecoli" else "normal",
        )
    ax.set_xlim(0.0, 1.48)
    ax.set_ylim(-0.5, 4.5)
    ax.set_title("Species relationship\n(schematic)", loc="left", fontsize=10.0)
    ax.axis("off")


def _draw_pr_inset(
    ax: plt.Axes,
    *,
    species: str,
    models: tuple[str, ...],
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    show_xlabel: bool,
) -> None:
    for model in models:
        curve = curves[(curves["model"] == model) & (curves["species"] == species)]
        ax.step(
            curve["recall"],
            curve["precision"],
            where="post",
            color=MODEL_COLORS[model],
            linestyle=MODEL_STYLES[model],
            linewidth=1.45,
            zorder=2,
        )
    prevalence = float(metrics.loc[metrics["species"] == species, "positive_fraction"].iloc[0])
    ax.axhline(
        prevalence,
        color="#777777",
        linestyle=(0, (3, 2)),
        linewidth=0.75,
        zorder=1,
    )
    _style_pr_axis(ax, show_xlabel=show_xlabel)


def _draw_aupr_bars(
    ax: plt.Axes,
    *,
    models: tuple[str, ...],
    metrics: pd.DataFrame,
) -> None:
    y_base = np.arange(len(DISPLAY_SPECIES) - 1, -1, -1, dtype=np.float64)
    if len(models) == 1:
        offsets = (0.0,)
        height = 0.46
    else:
        offsets = (0.16, -0.16)
        height = 0.27
    for model, offset in zip(models, offsets, strict=True):
        values = np.array(
            [
                metrics.loc[
                    (metrics["model"] == model) & (metrics["species"] == species),
                    "average_precision_aupr",
                ].iloc[0]
                for species in DISPLAY_SPECIES
            ],
            dtype=np.float64,
        )
        bars = ax.barh(
            y_base + offset,
            values,
            height=height,
            color=MODEL_COLORS[model],
            alpha=0.86 if model == "humanv11" else 0.72,
            edgecolor="none",
            label=model_label(model),
            zorder=2,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                min(0.985, value + 0.012),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left" if value < 0.96 else "right",
                va="center",
                fontsize=9.8,
                fontweight="semibold",
            )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.5, 4.5)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("Precision–Recall AUC (average precision)")
    ax.set_title("PPI prediction benchmarking", fontsize=15.0, pad=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#222222")
    if len(models) > 1:
        ax.legend(frameon=False, loc="lower right", title="Checkpoints")


def _plot_pr_benchmark(
    *,
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    figure_dir: Path,
    models: tuple[str, ...],
    title: str,
    stem_name: str,
) -> None:
    fig = plt.figure(figsize=(17.2, 10.2))
    grid = fig.add_gridspec(
        5,
        3,
        width_ratios=(1.75, 2.05, 7.7),
        left=0.045,
        right=0.985,
        bottom=0.115,
        top=0.87,
        wspace=0.17,
        hspace=0.28,
    )
    tree_ax = fig.add_subplot(grid[:, 0])
    _draw_species_tree(tree_ax)
    for row_index, species in enumerate(DISPLAY_SPECIES):
        pr_ax = fig.add_subplot(grid[row_index, 1])
        _draw_pr_inset(
            pr_ax,
            species=species,
            models=models,
            metrics=metrics,
            curves=curves,
            show_xlabel=row_index == len(DISPLAY_SPECIES) - 1,
        )
        if row_index == 0:
            pr_ax.set_title("PR curves", fontsize=11.0, pad=6)
    bar_ax = fig.add_subplot(grid[:, 2])
    _draw_aupr_bars(bar_ax, models=models, metrics=metrics)

    fig.text(0.03, 0.955, "b", fontsize=26, fontweight="bold", va="top")
    fig.suptitle(
        title,
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=19.0,
        fontweight="semibold",
    )
    fig.text(
        0.075,
        0.920,
        "Precision–recall curves and AUPRC for each species",
        ha="left",
        fontsize=10.3,
        color="#555555",
    )
    fig.text(
        0.075,
        0.035,
        "Dashed lines show positive prevalence. Species relationships are schematic.",
        ha="left",
        fontsize=9.2,
        color="#666666",
    )
    save_figure(fig, stem_name, figure_dir)


def plot_pr_figures(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Export reference-style species tree, PR insets, and AUPR bars."""
    configure_style()
    for model in MODEL_REPOSITORIES:
        _plot_pr_benchmark(
            metrics=metrics,
            curves=curves,
            figure_dir=figure_dir,
            models=(model,),
            title=f"{predictor_label()} {model_label(model)} PPI prediction benchmarking",
            stem_name=f"pr_auc_{model}",
        )
    _plot_pr_benchmark(
        metrics=metrics,
        curves=curves,
        figure_dir=figure_dir,
        models=tuple(MODEL_REPOSITORIES),
        title=f"{predictor_label()} checkpoint comparison",
        stem_name="pr_auc_checkpoint_comparison",
    )
