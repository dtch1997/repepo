import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from repepo.experiments.persona_generalization import (
    PersonaCrossSteeringExperimentResult,
    base_dataset_position,
)
from repepo.steering.evaluate_cross_steering import CrossSteeringResult
import numpy as np
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


def build_cross_steering_df(experiment_results):
    rows = []
    for dataset, result in experiment_results.items():
        layer = list(
            list(result.steering_vectors.values())[0].layer_activations.keys()
        )[0]
        cs = result.cross_steering_result
        multipliers = [
            *list(cs.neg_steering.keys()),
            0.0,
            *list(cs.pos_steering.keys()),
        ]
        for ds_index, ds_name in enumerate(cs.dataset_labels):
            self_sv_index = cs.steering_labels.index(ds_name)
            self_propensities_ld = get_propensities(
                cs, ds_index, self_sv_index, "mean_logit_diff"
            )
            self_propensities_prob = get_propensities(
                cs, ds_index, self_sv_index, "mean_pos_prob"
            )
            self_steerability_slope_ld = get_steerability_slope(
                np.array(multipliers), np.array([self_propensities_ld])
            )[0]
            self_steerability_slope_prob = get_steerability_slope(
                np.array(multipliers), np.array([self_propensities_prob])
            )[0]
            for sv_index, sv_name in enumerate(cs.steering_labels):
                propensities_ld = get_propensities(
                    cs, ds_index, sv_index, "mean_logit_diff"
                )
                propensities_prob = get_propensities(
                    cs, ds_index, sv_index, "mean_pos_prob"
                )

                steerability_slope_ld = get_steerability_slope(
                    np.array(multipliers), np.array([propensities_ld])
                )[0]
                steerability_slope_prob = get_steerability_slope(
                    np.array(multipliers), np.array([propensities_prob])
                )[0]

                if sv_name == "mean":
                    continue
                else:
                    train_ds_index = cs.dataset_labels.index(sv_name)
                    sv_base_prob = cs.dataset_baselines[train_ds_index].metrics[
                        "mean_pos_prob"
                    ]
                    sv_base_ld = cs.dataset_baselines[train_ds_index].metrics[
                        "mean_logit_diff"
                    ]

                base_prob = cs.dataset_baselines[ds_index].metrics["mean_pos_prob"]
                base_ld = cs.dataset_baselines[ds_index].metrics["mean_logit_diff"]
                pos_steered_ld = propensities_ld[-2]
                pos_steered_prob = propensities_prob[-2]

                dataset_type = ds_to_category.get(dataset, "other")
                base_pos = base_dataset_position(result, "raw")
                base_pos_js = base_dataset_position(result, "js")
                base_pos_logit = base_dataset_position(
                    result, "raw", metric_name="mean_logit_diff"
                )
                rel_base_pos = base_pos
                rel_base_pos_js = base_pos_js
                rel_base_pos_logit = base_pos_logit

                if ds_name == "baseline":
                    rel_base_pos = 1.0
                    rel_base_pos_js = 1.0
                    rel_base_pos_logit = 1.0
                elif "neg" in ds_name:
                    rel_base_pos = 1.0 - base_pos
                    rel_base_pos_js = 1.0 - base_pos_js
                    rel_base_pos_logit = 1.0 - base_pos_logit

                sv_base_lds = [
                    pred.metrics["logit_diff"]
                    for pred in cs.dataset_baselines[train_ds_index].predictions
                ]
                base_lds = [
                    pred.metrics["logit_diff"]
                    for pred in cs.dataset_baselines[ds_index].predictions
                ]
                ld_deltas = [abs(base - sv) for base, sv in zip(base_lds, sv_base_lds)]

                rows.append(
                    {
                        "dataset": dataset,
                        "steering_vec": sv_name,
                        "test_variation": ds_name,
                        "base_prob": base_prob,
                        "base_ld": base_ld,
                        "steerability_pos_delta_ld": pos_steered_ld - base_ld,
                        "steerability_pos_delta_prob": pos_steered_prob - base_prob,
                        "steerability_slope_ld": steerability_slope_ld,
                        # "steerability_spearman_ld": get_steerabilty_spearman(np.array(multipliers), np.array([propensities_ld])),
                        "steerability_slope_prob": steerability_slope_prob,
                        "self_steerability_slope_ld": self_steerability_slope_ld,
                        # "steerability_spearman_prob": get_steerabilty_spearman(np.array(multipliers), np.array([propensities_prob])),
                        "rel_steerability_slope_ld": steerability_slope_ld
                        / abs(self_steerability_slope_ld),
                        "rel_steerability_slope_ld_capped": max(
                            -1,
                            min(1, steerability_slope_ld / self_steerability_slope_ld),
                        ),
                        "rel_steerability_slope_prob": steerability_slope_prob
                        / abs(self_steerability_slope_prob),
                        "rel_steerability_slope_prob_capped": max(
                            -1,
                            min(
                                1,
                                steerability_slope_prob / self_steerability_slope_prob,
                            ),
                        ),
                        "base_pos": base_pos,
                        "base_pos_js": base_pos_js,
                        "rel_base_pos": rel_base_pos,
                        "rel_base_pos_js": rel_base_pos_js,
                        "rel_base_pos_logit": rel_base_pos_logit,
                        "ld_delta": abs(sv_base_ld - base_ld),
                        "ld_delta_per_ex_mean": mean(ld_deltas),
                        "prob_delta": abs(sv_base_prob - base_prob),
                        "dataset_type": dataset_type,
                    }
                )
    return pd.DataFrame.from_records(rows)


def plot_persona_relative_steerability_trends(
    results: dict[str, PersonaCrossSteeringExperimentResult],
    title: str,
    figsize=(10, 4),
    save_path: str | None = None,
    min_self_steerability: float = 0.25,
):
    cs_df = build_cross_steering_df(results)
    cs_df = cs_df[cs_df["self_steerability_slope_ld"] > min_self_steerability]
    x = "ld_delta_per_ex_mean"
    y = "rel_steerability_slope_ld"

    plt.figure(figsize=figsize)

    # Create the scatter plot
    scatter_plot = sns.scatterplot(data=cs_df, x=x, y=y)  # type: ignore

    # Calculate the linear fit line
    spearman_corr, _ = stats.spearmanr(cs_df[x], cs_df[y])
    print(f"Spearman's Correlation: {spearman_corr}")

    # Add the linear fit line
    sns.regplot(data=cs_df, x=x, y=y, scatter=False, ci=None)  # type: ignore

    plt.title(title)
    scatter_plot.set(
        xlabel="Unsteered LD delta between training datasets",
        ylabel="Relative steerability",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_rel_steerabilities(
    cs_df,
    ax,
    title,
):
    x = "ld_delta_per_ex_mean"
    y = "rel_steerability_slope_ld"

    # Create the scatter plot
    sns.scatterplot(data=cs_df, x=x, y=y, ax=ax)

    # Calculate the linear fit line
    spearman_corr, _ = stats.spearmanr(cs_df[x], cs_df[y])
    print(f"Spearman's Correlation: {spearman_corr}")

    # Add the linear fit line
    sns.regplot(data=cs_df, x=x, y=y, ax=ax, scatter=False)

    ax.axhline(0, color="black", linestyle="-")
    ax.axvline(0, color="black", linestyle="-")
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Relative steerability", fontsize=16)
    ax.set_xlabel(None)


def plot_persona_relative_steerability_trends_dual(
    results_left: dict[str, PersonaCrossSteeringExperimentResult],
    results_right: dict[str, PersonaCrossSteeringExperimentResult],
    label_left: str,
    label_right: str,
    figsize=(9, 3.5),
    save_path: str | None = None,
    min_self_steerability: float = 0.25,
):
    sns.set_theme()
    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)

    cs_df_left = build_cross_steering_df(results_left)
    cs_df_left = cs_df_left[
        cs_df_left["self_steerability_slope_ld"] > min_self_steerability
    ]

    cs_df_right = build_cross_steering_df(results_right)
    cs_df_right = cs_df_right[
        cs_df_right["self_steerability_slope_ld"] > min_self_steerability
    ]

    plot_rel_steerabilities(cs_df_left, title=label_left, ax=axs[0])
    plot_rel_steerabilities(cs_df_right, title=label_right, ax=axs[1])

    fig.supxlabel("Unsteered LD delta between training datasets", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
