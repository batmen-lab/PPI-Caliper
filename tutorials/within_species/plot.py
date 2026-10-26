"""Within-species FDP, power, and accuracy figures."""

import matplotlib.pyplot as plt
import pandas as pd

from utils.figure_style import DISPLAY_SPECIES, MODELS, SPECIES_LABELS, draw_mean_curve


def plot_results(results: pd.DataFrame, predictor: str) -> plt.Figure:
    frame = results.assign(
        power=results.true_positives / results.n_test_positive,
        accuracy=(
            results.true_positives + results.n_test - results.n_test_positive
            - results.false_positives
        ) / results.n_test,
    )
    metrics = (
        ("observed_FDP", "Mean held-out FDP"),
        ("power", "Mean power"),
        ("accuracy", "Mean accuracy"),
    )
    figure, axes = plt.subplots(
        3, 5, figsize=(14, 8), sharex=True, sharey=True, layout="constrained"
    )
    for row, (metric, label) in enumerate(metrics):
        for column, species in enumerate(DISPLAY_SPECIES):
            axis = axes[row, column]
            if row == 0:
                axis.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1)
                axis.set_title(SPECIES_LABELS[species])
            for model in MODELS:
                subset = frame[(frame.model == model) & (frame.species == species)]
                draw_mean_curve(axis, subset, metric, model)
            axis.set(xlim=(0, 1), ylim=(0, 1), xticks=[0.1, 0.3, 0.5, 0.7, 0.9])
            if row == 2:
                axis.set_xlabel("Target FDR")
        axes[row, 0].set_ylabel(label)
    axes[0, -1].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle(f"{predictor}: within-species calibration, {frame.seed.nunique()} splits")
    return figure
