"""Bar charts and figure exports shared by the manuscript plotting modules."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import t as student_t

SPECIES_NAMES = {
    "mouse": "Mouse", "fly": "Fly", "worm": "Worm", "yeast": "Yeast", "ecoli": "E. coli",
}
PREDICTOR_NAMES = {"plm-interact": "PLM-Interact", "mint": "MINT"}
MODELS = ("humanv11", "humanv12")
TARGET_COLORS = LinearSegmentedColormap.from_list("target_fdr", [
    "#281e41", "#2d285c", "#2e4282", "#2a6491", "#2d8698",
    "#3bac9c", "#60cba2", "#83d7ad", "#a6e3bc",
])


def draw_bars(ax, rows, metric):
    """Plot the split mean and its 95% interval for each target FDR."""
    stats = rows.groupby("target_fdr")[metric].agg(["mean", "sem", "count"])
    q = stats.index.to_numpy()
    mean = stats["mean"]
    critical = student_t.ppf(0.975, stats["count"] - 1) if metric == "Empirical FDR" else 1.96
    margin = (critical * stats["sem"]).fillna(0)
    lower = (mean - margin).clip(lower=0)
    upper = (mean + margin).clip(upper=1)

    if metric == "Empirical FDR":
        ax.plot(q, q, "--", color="black", linewidth=0.9, zorder=1)
    colors = TARGET_COLORS(np.interp(q, [0.1, 0.9], [0.0, 1.0]))
    width = 0.72 * np.diff(q).min(initial=0.1)
    ax.bar(q, mean, width=width, color=colors, edgecolor="black", linewidth=0.5, zorder=2)
    ax.errorbar(q, mean, yerr=[mean - lower, upper - mean], fmt="none",
                ecolor="black", elinewidth=0.65, capsize=1.8, capthick=0.65, zorder=3)
    for level, value, top in zip(q, mean, upper):
        ax.text(level, top + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=6.8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1})
    ax.set(xlim=(min(0.03, q[0] - 0.07), max(0.97, q[-1] + 0.07)),
           ylim=(0, 1.1), yticks=np.arange(0, 1.01, 0.2))
    ax.set_xticks(q, [f"{level:g}" for level in q], rotation=40, ha="right")
    ax.tick_params(labelsize=7, length=2.5)
    ax.set_box_aspect(1)
    ax.grid(False)


def save_figure(figure, name, output_dir) -> None:
    """Save a PNG and a vector PDF with embedded TrueType fonts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    with plt.rc_context({"pdf.fonttype": 42}):
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
