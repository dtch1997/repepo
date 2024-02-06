import matplotlib.pyplot as plt
import json
import pyrallis

from repepo.experiments.caa_repro.utils.helpers import (
    get_experiment_path,
)
from repepo.experiments.evaluate_tqa_caa import EvaluateTqaCaaConfig, EvaluateCaaResult


def plot_in_distribution_data_for_layer(
    ax: plt.Axes,
    all_results: list[EvaluateCaaResult],
    title: str,
    layers: list[int],
    linestyle="dashed",
    label_suffix="",
) -> plt.Axes:
    # Create figure

    viridis = plt.cm.get_cmap("viridis", len(layers))
    for layer in layers:
        layer_results = [x for x in all_results if x.layer_id == layer]
        layer_results.sort(key=lambda x: x.multiplier)
        ax.plot(
            [x.multiplier for x in layer_results],
            [x.average_key_prob for x in layer_results],
            label=f"Layer {layer} {label_suffix}",
            marker="o",
            linestyle=linestyle,
            markersize=5,
            linewidth=2.5,
            # color by layer,
            color=viridis(layers.index(layer)),
        )

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Probability of sycophantic answer to A/B question")
    return ax


if __name__ == "__main__":
    config = pyrallis.parse(EvaluateTqaCaaConfig)
    save_suffix = config.settings.make_result_save_suffix(None, None)

    experiment_name = (
        f"{config.train_dataset_spec.name}_{config.test_dataset_spec.name}"
    )
    results_path = get_experiment_path() / experiment_name / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"results_{save_suffix}.json", "r") as f:
        _results = json.load(f)
        results = [EvaluateCaaResult(**res) for res in _results]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax = plot_in_distribution_data_for_layer(
        ax, results, experiment_name, config.layers
    )
    fig.tight_layout()
    fig.savefig(results_path / f"results_{save_suffix}.png")
