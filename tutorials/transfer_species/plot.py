"""Cross-species FDP panels and AUPRC trees for manuscript Figures 3 and A4–A6."""

import matplotlib.pyplot as plt

from utils.figure_style import MODELS, PREDICTOR_NAMES, SPECIES_NAMES, draw_bars, save_figure


FIGURE_NUMBERS = {
    ("plm-interact", "humanv11"): "3",
    ("plm-interact", "humanv12"): "A4",
    ("mint", "humanv11"): "A5",
    ("mint", "humanv12"): "A6",
}


def _draw_species_pair(ax, transfer, within, source, target):
    """Draw one checkpoint's results, using the within-species experiment on the diagonal."""
    if source == target:
        rows = within.loc[within.species == source, ["target_fdr", "observed_FDP"]]
    else:
        rows = transfer.loc[
            (transfer.source_species == source) & (transfer.target_species == target),
            ["target_fdr", "target_observed_FDP"],
        ]
    rows.columns = ["target_fdr", "Empirical FDR"]
    draw_bars(ax, rows, "Empirical FDR")


def _draw_species_tree(ax, auprc, row_margin):
    """Annotate the species relationships with full-table average precision."""
    def join(x, left, right):
        x1, y1 = left
        x2, y2 = right
        ax.plot([x1, x, x, x2], [y1, y1, y2, y2], color="0.5", linewidth=1.3)
        return x, (y1 + y2) / 2

    tips = {species: (0.72, 4 - row) for row, species in enumerate(SPECIES_NAMES)}
    fly_worm = join(0.58, tips["fly"], tips["worm"])
    animals = join(0.39, tips["mouse"], fly_worm)
    eukaryotes = join(0.21, animals, tips["yeast"])
    join(0.04, eukaryotes, tips["ecoli"])
    for species, (x, y) in tips.items():
        ax.text(x + 0.06, y + 0.07, f"{auprc[species]:.2f}", va="bottom", fontsize=9)
        ax.text(x + 0.06, y, SPECIES_NAMES[species], va="center", fontsize=10)
    ax.set(xlim=(0, 1.48), ylim=(-row_margin, 4 + row_margin))
    ax.set_title("Model performance\nAUPRC", loc="left", fontsize=10)
    ax.set_axis_off()


def plot_transfer_figure(transfer, within, classification, predictor, model):
    """Build a manuscript grid from a predictor's transfer, within-species and classification tables."""
    transfer = transfer.loc[transfer.model == model]
    within = within.loc[within.model == model]
    auprc = classification.loc[classification.model == model].set_index("species")["average_precision_aupr"]

    fig = plt.figure(figsize=(15, 14.1))
    row_spacing = 0.24
    grid = fig.add_gridspec(
        5, 6, width_ratios=[0.74, 1, 1, 1, 1, 1],
        left=0.02, right=0.99, bottom=0.055, top=0.91,
        hspace=row_spacing, wspace=0.12,
    )
    _draw_species_tree(fig.add_subplot(grid[:, 0]), auprc, row_margin=0.5 / (1 + row_spacing))
    for row, source in enumerate(SPECIES_NAMES):
        for col, target in enumerate(SPECIES_NAMES):
            ax = fig.add_subplot(grid[row, col + 1])
            _draw_species_pair(ax, transfer, within, source, target)
            if row == 0:
                ax.set_title(SPECIES_NAMES[target], weight="semibold")
            if col == 0:
                ax.set_ylabel("Empirical FDR")
    number = FIGURE_NUMBERS[predictor, model]
    fig.suptitle(f"Figure {number} · {PREDICTOR_NAMES[predictor]} {model}", weight="semibold")
    fig.text(0.5, 0.95, "Rows show calibration species · Columns show test species · "
             f"{transfer.seed.nunique()} splits", ha="center")
    fig.supxlabel("Target FDR", y=0.014)
    return fig


def plot_transfer_panel(transfer, within, predictor, model, source, target):
    """Draw one source-to-target panel from the raw seed tables and return its figure."""
    fig, ax = plt.subplots(figsize=(4.3, 4), layout="constrained")
    _draw_species_pair(ax, transfer.loc[transfer.model == model], within.loc[within.model == model],
                       source, target)
    ax.set(
        xlabel=f"Target FDR for calibration on {SPECIES_NAMES[source]}",
        ylabel=f"Mean held-out FDP on {SPECIES_NAMES[target]}",
        title=f"{PREDICTOR_NAMES[predictor]} {model} · {SPECIES_NAMES[source]} → {SPECIES_NAMES[target]}",
    )
    return fig


def save_transfer_figures(results, classification, predictor, figure_dir):
    """Write each checkpoint grid and all 25 directional panels as PNG and PDF."""
    # Every target repeats the same source holdout for a given checkpoint, seed and q.
    # Take it once for each diagonal panel, so this driver can also run on its own.
    within = (
        results[["model", "source_species", "seed", "target_fdr", "source_holdout_observed_FDP"]]
        .drop_duplicates(subset=["model", "source_species", "seed", "target_fdr"])
        .rename(columns={"source_species": "species", "source_holdout_observed_FDP": "observed_FDP"})
    )
    for model in MODELS:
        grid = plot_transfer_figure(results, within, classification, predictor, model)
        number = FIGURE_NUMBERS[predictor, model].lower()
        save_figure(grid, f"figure_{number}_{predictor}_{model}", figure_dir)
        for source in SPECIES_NAMES:
            for target in SPECIES_NAMES:
                panel = plot_transfer_panel(results, within, predictor, model, source, target)
                save_figure(panel, f"{source}_to_{target}", figure_dir / "panels" / model)
