import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from steering_vectors import SteeringVector
from matplotlib import patheffects


def plot_steering_vector_cos_similarities(
    named_steering_vectors: dict[str, SteeringVector],
    layer: int,
    title: str,
    save_path: str | None = None,
):
    named_raw_vectors = {
        key: sv.layer_activations[layer] for key, sv in named_steering_vectors.items()
    }
    named_norm_vectors = {
        key: vec / vec.norm() for key, vec in named_raw_vectors.items()
    }
    names = list(named_norm_vectors.keys())
    norm_vectors = torch.stack([named_norm_vectors[key] for key in names])

    cos_sims = norm_vectors @ norm_vectors.T
    cos_sims_list = cos_sims.tolist()

    sns.heatmap(
        cos_sims,
        center=0,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )

    # Iterate over the data and create a text annotation for each cell
    for i in range(len(cos_sims_list)):
        for j in range(len(cos_sims_list)):
            text = plt.text(
                j + 0.5,
                i + 0.5,
                round(cos_sims_list[i][j], 3),
                ha="center",
                va="center",
                color="w",
                path_effects=[
                    patheffects.withStroke(linewidth=2, foreground="#33333370")
                ],
            )

    # Add a colorbar to show the scale
    # plt.colorbar()
    plt.title(f"{title}")
    plt.xticks(ticks=np.arange(len(names)) + 0.5, labels=names)
    plt.yticks(ticks=np.arange(len(names)) + 0.5, labels=names)
    plt.xlabel("Steering vector")
    plt.ylabel("Steering vector")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()
