import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from repepo.steering.evaluate_cross_steering import (
    CrossSteeringResult,
)
from matplotlib import patheffects


def plot_cross_steering_result(
    result: CrossSteeringResult, title: str, save_path: str | None = None
):
    deltas = []
    for baseline, dataset_pos_steering in zip(
        result.dataset_baseline, result.pos_steering
    ):
        ds_deltas = [steering.mean - baseline.mean for steering in dataset_pos_steering]
        deltas.append(ds_deltas)

    deltas_tensor = torch.tensor(deltas)
    largest_abs_val = deltas_tensor.abs().max().item()
    sns.heatmap(
        deltas_tensor,
        center=0,
        cmap="RdBu_r",
        vmin=-1 * largest_abs_val,
        vmax=largest_abs_val,
    )

    # Iterate over the data and create a text annotation for each cell
    for i in range(len(deltas)):
        for j in range(len(deltas)):
            text = plt.text(
                j + 0.5,
                i + 0.5,
                round(deltas[i][j], 3),
                ha="center",
                va="center",
                color="w",
                path_effects=[
                    patheffects.withStroke(linewidth=2, foreground="#33333370")
                ],
            )

    ds_labels = result.dataset_labels
    sv_labels = result.steering_labels

    # Add a colorbar to show the scale
    # plt.colorbar()
    plt.title(f"{title} (pos prob delta)")
    plt.xticks(ticks=np.arange(len(sv_labels)) + 0.5, labels=sv_labels)
    plt.yticks(ticks=np.arange(len(ds_labels)) + 0.5, labels=ds_labels)
    plt.xlabel("Steering vector")
    plt.ylabel("Dataset")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()
