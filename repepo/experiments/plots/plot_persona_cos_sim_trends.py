import pandas as pd
from statistics import mean

import torch
from repepo.experiments.persona_generalization import (
    PersonaCrossSteeringExperimentResult,
)
from repepo.utils.stats import bernoulli_js_dist
from repepo.experiments.persona_prompts import CATEGORIZED_PERSONA_PROMPTS

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def plot_persona_cos_sim_trends(
    results: dict[str, PersonaCrossSteeringExperimentResult],
    title: str,
    show_fit_line: bool = True,
    figsize=(10, 4),
    save_path: str | None = None,
):
    df = build_pairs_df(results)
    dist_col = "logit_dist"
    plt.figure(figsize=figsize)

    # Create the scatter plot
    scatter_plot = sns.scatterplot(data=df, x=dist_col, y="cos", legend="full")

    # Calculate the linear fit line
    spearman_corr, _ = stats.spearmanr(df[dist_col], df["cos"])
    print(f"Spearman's Correlation: {spearman_corr}")

    if show_fit_line:
        # Add the linear fit line
        sns.regplot(
            data=df,
            x=dist_col,
            y="cos",
            scatter=False,
            ci=None,
            line_kws={"color": "red"},
        )

    scatter_plot.set(xlabel="Unsteered LD delta between training datasets")

    plt.title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()


ds_to_category = {}
for category, dataset_prompts in CATEGORIZED_PERSONA_PROMPTS.items():
    for dataset in dataset_prompts.keys():
        ds_to_category[dataset] = category


def build_pairs_df(
    results: dict[str, PersonaCrossSteeringExperimentResult],
):
    rows = []
    for dataset, result in results.items():
        layer = list(
            list(result.steering_vectors.values())[0].layer_activations.keys()
        )[0]
        cs = result.cross_steering_result
        named_raw_vectors = {
            key: sv.layer_activations[layer]
            for key, sv in result.steering_vectors.items()
        }
        named_norm_vectors = {
            key: vec / vec.norm() for key, vec in named_raw_vectors.items()
        }
        names = list(named_norm_vectors.keys())
        norm_vectors = torch.stack([named_norm_vectors[key] for key in names])

        cos_sims = (norm_vectors @ norm_vectors.T).tolist()
        for i, name1 in enumerate(names):
            for j in range(i + 1, len(names)):
                name2 = names[j]
                if name1 == "mean":
                    base_prob1 = mean(
                        [bs.metrics["mean_pos_prob"] for bs in cs.dataset_baselines]
                    )
                    base_logits1 = mean(
                        [bs.metrics["mean_logit_diff"] for bs in cs.dataset_baselines]
                    )
                else:
                    base_prob1 = cs.dataset_baselines[
                        cs.dataset_labels.index(name1)
                    ].metrics["mean_pos_prob"]
                    base_logits1 = cs.dataset_baselines[
                        cs.dataset_labels.index(name1)
                    ].metrics["mean_logit_diff"]
                if name2 == "mean":
                    base_prob2 = mean(
                        [bs.metrics["mean_pos_prob"] for bs in cs.dataset_baselines]
                    )
                    base_logits2 = mean(
                        [bs.metrics["mean_logit_diff"] for bs in cs.dataset_baselines]
                    )
                else:
                    base_prob2 = cs.dataset_baselines[
                        cs.dataset_labels.index(name2)
                    ].metrics["mean_pos_prob"]
                    base_logits2 = cs.dataset_baselines[
                        cs.dataset_labels.index(name2)
                    ].metrics["mean_logit_diff"]
                pair = f"{name1}-{name2}"
                if "pos" in pair and "neg" in pair:
                    pairing_type = "pos-neg"
                elif "pos" in pair and "base" in pair:
                    pairing_type = "base-pos"
                elif "pos" in pair and "mean" in pair:
                    pairing_type = "mean-pos"
                elif "neg" in pair and "mean" in pair:
                    pairing_type = "mean-neg"
                elif pair.count("neg") == 2:
                    pairing_type = "neg-neg"
                elif pair.count("pos") == 2:
                    pairing_type = "pos-pos"
                else:
                    pairing_type = "base-neg"

                if "SYS" in pair and "PT" in pair:
                    prompts_type = "sys-pt"
                elif "SYS" in pair and "base" in pair:
                    prompts_type = "base-sys"
                elif "SYS" in pair and "mean" in pair:
                    prompts_type = "mean-sys"
                elif "PT" in pair and "mean" in pair:
                    prompts_type = "mean-pt"
                elif pair.count("PT") == 2:
                    prompts_type = "pt-pt"
                elif pair.count("SYS") == 2:
                    prompts_type = "sys-sys"
                else:
                    prompts_type = "base-pt"

                dataset_type = ds_to_category.get(dataset, "other")
                rows.append(
                    {
                        "dataset": dataset,
                        "pair": pair,
                        "cos": cos_sims[i][j],
                        "js_dist": abs(bernoulli_js_dist(base_prob1, base_prob2)),
                        "prob_dist": abs(base_prob1 - base_prob2),
                        "logit_dist": abs(base_logits1 - base_logits2),
                        "pairing_type": pairing_type,
                        "prompts_type": prompts_type,
                        "dataset_type": dataset_type,
                        "overall": "overall",  # hacky way to re-use grouping code
                    }
                )
    return pd.DataFrame.from_records(rows)
