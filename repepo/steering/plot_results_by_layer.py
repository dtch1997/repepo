import matplotlib.pyplot as plt
from repepo.steering.utils.helpers import SteeringConfig, SteeringResult


def plot_results_by_layer(
    ax: plt.Axes, config: SteeringConfig, results: list[SteeringResult]
):
    for i, layer in enumerate(config.layers):
        layer_results = [x for x in results if x.layer_id == layer]
        layer_results.sort(key=lambda x: x.multiplier)

        ax.plot(
            [x.multiplier for x in layer_results],
            [x.logit_diff for x in layer_results],
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
            label=f"Layer {layer}",
        )

    ax.set_title(f"{config.train_dataset_name}")
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Mean logit difference")
    ax.legend()
    return ax
