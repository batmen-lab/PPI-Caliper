"""Shared colors, confidence bands, and figure output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from utils.inputs import SPECIES_LABELS

DISPLAY_SPECIES = ("mouse", "fly", "worm", "yeast", "ecoli")
MODELS = ("humanv11", "humanv12")
MODEL_COLORS = {"humanv11": "#285E85", "humanv12": "#B66A32"}
MODEL_STYLES = {"humanv11": "-", "humanv12": "--"}


def draw_mean_curve(axis, frame, metric, model):
    group = frame.groupby("target_fdr")[metric].agg(["mean", "std", "count"])
    mean = group["mean"]
    margin = (
        student_t.ppf(0.975, group["count"] - 1)
        * group["std"] / np.sqrt(group["count"])
    )
    axis.plot(
        group.index, mean,
        color=MODEL_COLORS[model], linestyle=MODEL_STYLES[model],
        marker="o", markersize=3, label=model,
    )
    axis.fill_between(
        group.index, (mean - margin).clip(0, 1), (mean + margin).clip(0, 1),
        color=MODEL_COLORS[model], alpha=0.1,
    )


def save_figure(figure, name, output_dir) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def show_plot(figure, name, output_dir):
    from IPython.display import Image, display

    path = save_figure(figure, name, output_dir)
    display(Image(filename=str(path), alt=name.replace("_", " ")))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.axisbelow": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 7,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
