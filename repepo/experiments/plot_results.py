import matplotlib.pyplot as plt
import json
import pyrallis

from repepo.experiments.caa_repro.utils.helpers import (
    get_experiment_path,
)
from repepo.experiments.evaluate_tqa_caa import EvaluateTqaCaaConfig, EvaluateCaaResult


def plot_in_distribution_data_for_layer(
    all_results: list[EvaluateCaaResult],
    layers: list[int],
) -> plt.Figure:
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    for layer in layers:
        layer_results = [x for x in all_results if x.layer_id == layer]
        layer_results.sort(key=lambda x: x.multiplier)
        ax.plot(
            [x.multiplier for x in layer_results],
            [x.average_key_prob for x in layer_results],
            label=f"Layer {layer}",
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
        )

    ax.legend()
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Probability of sycophantic answer to A/B question")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    config = pyrallis.parse(EvaluateTqaCaaConfig)
    save_suffix = config.settings.make_result_save_suffix(None, None)

    if config.experiment_name is None:
        config.experiment_name = (
            f"{config.train_dataset_spec.name}_{config.test_dataset_spec.name}"
        )

    results_path = get_experiment_path(config.experiment_name) / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"results_{save_suffix}.json", "r") as f:
        _results = json.load(f)
        results = [EvaluateCaaResult(**res) for res in _results]

    fig = plot_in_distribution_data_for_layer(results, config.layers)
    fig.savefig(results_path / f"results_{save_suffix}.png")
