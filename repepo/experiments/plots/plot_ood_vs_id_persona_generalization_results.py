import pandas as pd
from repepo.experiments.persona_generalization import (
    PersonaCrossSteeringExperimentResult,
)
from repepo.steering.evaluate_cross_steering import CrossSteeringResult
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
from repepo.steering.steerability import (
    get_steerability_slope,
)
from repepo.experiments.persona_prompts import CATEGORIZED_PERSONA_PROMPTS


ds_to_category = {}
for category, dataset_prompts in CATEGORIZED_PERSONA_PROMPTS.items():
    for dataset in dataset_prompts.keys():
        ds_to_category[dataset] = category


def get_propensities(
    cs: CrossSteeringResult, ds_index: int, sv_index: int, metric_name: str
):
    return [
        *[
            res[ds_index][sv_index].metrics[metric_name]
            for res in cs.neg_steering.values()
        ],
        cs.dataset_baselines[ds_index].metrics[metric_name],
        *[
            res[ds_index][sv_index].metrics[metric_name]
            for res in cs.pos_steering.values()
        ],
    ]


def plot_ood_vs_id_persona_generalization_results(
    results: dict[str, PersonaCrossSteeringExperimentResult],
    title: str = "OOD vs ID steerability",
    save_path: str | None = None,
):
    rows = []
    for dataset, result in results.items():
        cs = result.cross_steering_result
        multipliers = [
            *list(cs.neg_steering.keys()),
            0.0,
            *list(cs.pos_steering.keys()),
        ]
        baseline_sv_index = cs.steering_labels.index("baseline")
        baseline_ds_index = cs.dataset_labels.index("baseline")
        id_propensities_ld = get_propensities(
            cs, baseline_ds_index, baseline_sv_index, "mean_logit_diff"
        )
        id_steerability = get_steerability_slope(
            np.array(multipliers), np.array([id_propensities_ld])
        )[0]
        ood_steerabilities = []
        for ds_index, ds_name in enumerate(cs.dataset_labels):
            if ds_name == "baseline" or ds_name == "mean":
                continue
            propensities_ld = get_propensities(
                cs, ds_index, baseline_sv_index, "mean_logit_diff"
            )
            steerability = get_steerability_slope(
                np.array(multipliers), np.array([propensities_ld])
            )[0]
            ood_steerabilities.append(steerability)
        ood_steerability = mean(ood_steerabilities)
        rows.append(
            {
                "dataset": dataset,
                "id_steerability": id_steerability,
                "ood_steerability": ood_steerability,
            }
        )
    df = pd.DataFrame.from_records(rows)

    sns.set_theme(style="darkgrid")
    ax = sns.scatterplot(
        data=df,
        x="id_steerability",
        y="ood_steerability",
    )
    ax.set_title(title)
    ax.set(
        xlabel="In-distribution steerability", ylabel="Out-of-distribution steerability"
    )

    maxes = [df["id_steerability"].max(), df["ood_steerability"].max()]
    mins = [df["id_steerability"].min(), df["ood_steerability"].min()]

    ax.set_xlim(int(min(mins)) - 0.25, round(max(maxes)) + 0.25)
    ax.set_ylim(int(min(mins)) - 0.25, round(max(maxes)) + 0.25)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
