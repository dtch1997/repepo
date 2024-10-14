from typing import Any

import pandas as pd
from repepo.steering.plots.utils import truncate_name
from repepo.steering.sweep_layers import SweepLayersResult
import seaborn as sns
import matplotlib.pyplot as plt


def plot_sweep_layers_result(
    result: SweepLayersResult,
    title: str = "Steerability by layer",
    save_path: str | None = None,
):
    steerabilities = result.steerabilities
    layers = result.layers
    df_data: dict[str, Any] = {"Layer": layers}

    sns.set_theme(style="darkgrid")

    for dataset, layer_steerabilities in steerabilities.items():
        df_data[truncate_name(dataset)] = [
            layer_steerabilities[layer] for layer in layers
        ]

    df = pd.DataFrame(df_data)
    df = df.melt("Layer", var_name="Dataset", value_name="Steerability")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Layer", y="Steerability", hue="Dataset")

    plt.title(title)
    plt.legend(title="Dataset", loc="lower left", fontsize="small")
    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    # Show the plot
    plt.show()

    return df