from typing import Any, Literal
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from repepo.steering.evaluate_cross_steering import (
    CrossSteeringResult,
)
from matplotlib import patheffects

from repepo.utils.stats import bernoulli_js_dist


DeltaType = Literal["pos_base", "base_neg", "pos_neg"]
DistMetric = Literal["js", "raw"]


def _calc_deltas(
    result: CrossSteeringResult,
    delta_type: DeltaType,
    dist_metric: DistMetric,
    metric_name: str,
) -> list[list[float]]:
    dist_fn = bernoulli_js_dist if dist_metric == "js" else lambda x, y: y - x
    deltas = []
    for baseline, dataset_pos_steering, dataset_neg_steering in zip(
        result.dataset_baselines, result.pos_steering, result.neg_steering
    ):
        if delta_type == "pos_base":
            ds_deltas = [
                dist_fn(baseline.metrics[metric_name], steering.metrics[metric_name])
                for steering in dataset_pos_steering
            ]
        elif delta_type == "base_neg":
            ds_deltas = [
                dist_fn(steering.metrics[metric_name], baseline.metrics[metric_name])
                for steering in dataset_neg_steering
            ]
        elif delta_type == "pos_neg":
            ds_deltas = [
                dist_fn(
                    neg_steering.metrics[metric_name], pos_steering.metrics[metric_name]
                )
                for pos_steering, neg_steering in zip(
                    dataset_pos_steering, dataset_neg_steering
                )
            ]
        deltas.append(ds_deltas)
    return deltas


def plot_cross_steering_result(
    result: CrossSteeringResult,
    title: str,
    delta_type: DeltaType = "pos_base",
    dist_metric: DistMetric = "raw",
    metric_name: str = "mean_pos_prob",
    save_path: str | None = None,
):
    deltas = _calc_deltas(
        result, delta_type, dist_metric=dist_metric, metric_name=metric_name
    )

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
        for j in range(len(deltas[i])):
            # for some reason round() doesn't type check with float??
            delta: Any = deltas[i][j]
            plt.text(
                j + 0.5,
                i + 0.5,
                round(delta, 3),
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
    plt.title(f"{title} ({dist_metric}, {delta_type})")
    plt.xticks(ticks=np.arange(len(sv_labels)) + 0.5, labels=sv_labels)
    plt.yticks(ticks=np.arange(len(ds_labels)) + 0.5, labels=ds_labels)
    plt.xlabel("Steering vector")
    plt.ylabel("Dataset")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()
