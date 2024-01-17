from repepo.experiments.caa_repro.utils.helpers import (
    get_experiment_path,
    SteeringSettings,
)
from dataclasses import dataclass
import pyrallis
from pyrallis import field
import matplotlib.pyplot as plt
import json

results_path = get_experiment_path() / "results"
analysis_path = get_experiment_path() / "analysis"


def plot_in_distribution_data_for_layer(
    layers: list[int], multipliers: list[float], settings: SteeringSettings
):
    save_suffix = settings.make_result_save_suffix()
    with open(results_path / f"results_{save_suffix}.json", "r") as f:
        all_results = json.load(f)

    # few_shot_options = [
    #     ("positive", "Sycophantic few-shot prompt"),
    #     ("negative", "Non-sycophantic few-shot prompt"),
    #     ("none", "No few-shot prompt"),
    # ]
    # settings.few_shot = None
    save_to = str(analysis_path / f"{settings.make_result_save_suffix()}.svg")

    # Create figure
    plt.clf()
    plt.figure(figsize=(6, 6))

    for layer in layers:
        layer_results = [x for x in all_results if x["layer_id"] == layer]
        layer_results.sort(key=lambda x: x["multiplier"])
        plt.plot(
            [x["multiplier"] for x in layer_results],
            [x["average_key_prob"] for x in layer_results],
            label=f"Layer {layer}",
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
        )

    # all_results = {}
    # for few_shot, label in few_shot_options:
    #     settings.few_shot = few_shot
    #     try:
    #         res_list = []
    #         for multiplier in multipliers:
    #             # results = get_data(layer, multiplier, settings)
    #             # avg_key_prob = get_avg_key_prob(results, "answer_matching_behavior")

    #             res_list.append((multiplier, avg_key_prob))
    #         res_list.sort(key=lambda x: x[0])
    #         plt.plot(
    #             [x[0] for x in res_list],
    #             [x[1] for x in res_list],
    #             label=label,
    #             marker="o",
    #             linestyle="dashed",
    #             markersize=5,
    #             linewidth=2.5,
    #         )
    #         all_results[few_shot] = res_list
    #     except:
    #         print(f"[WARN] Missing data for few_shot={few_shot} for layer={layer}")
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of sycophantic answer to A/B question")
    plt.tight_layout()
    # plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")

    # Save data in all_results used for plotting as .txt
    # with open(save_to.replace(".svg", ".txt"), "w") as f:
    #     for few_shot, res_list in all_results.items():
    #         for multiplier, score in res_list:
    #             f.write(f"{few_shot}\t{multiplier}\t{score}\n")


@dataclass
class PlotResultsConfig:
    """
    A single training sample for a steering vector.
    """

    layers: list[int] = field(default=[], is_mutable=True)
    multipliers: list[float] = field(default=[], is_mutable=True)
    settings: SteeringSettings = field(default=SteeringSettings(), is_mutable=True)


if __name__ == "__main__":
    analysis_path.mkdir(parents=True, exist_ok=True)
    config = pyrallis.parse(PlotResultsConfig)
    plot_in_distribution_data_for_layer(
        config.layers, config.multipliers, config.settings
    )
