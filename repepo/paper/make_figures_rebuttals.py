# %%
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
from repepo.paper.preprocess_results import compute_steerability
from repepo.paper.helpers import dataset_full_names_to_short_names

sns.set_theme()

# model = "qwen"
model = "llama7b"
df = pd.read_parquet(f"{model}_steerability.parquet.gzip")
df["dataset_name"] = df["dataset_name"].replace(dataset_full_names_to_short_names)
with open("selected_datasets.json") as f:
    selected_datasets = json.load(f)


# %%
# Barplot of the difference in steerability for several randomly selected datasets


def make_plot_for_multiplier_ablations(df, selected_datasets: list[str] | None = None):
    # If not configured, use all datasets
    if selected_datasets is None:
        selected_datasets = df.dataset_name.unique()[:5]
    assert selected_datasets is not None

    dfs = []

    for dataset in selected_datasets:
        # Only do the in-distribution results
        subset_df = df[
            (df.dataset_name == dataset)
            & (df.steering_label == "baseline")
            & (df.dataset_label == "baseline")
        ]
        # Remove previous results
        subset_df = subset_df[
            [
                "logit_diff",
                "multiplier",
                "dataset_name",
                "steering_label",
                "dataset_label",
                "test_example.idx",
            ]
        ]
        assert len(subset_df) > 0
        for max_multiplier in (0.5, 1.0, 1.5):
            steerability = compute_steerability(
                subset_df, multiplier_range=(-max_multiplier, max_multiplier)
            ).drop_duplicates()
            steerability = steerability[["dataset_name", "slope"]]
            steerability["max_multiplier"] = max_multiplier
            dfs.append(steerability)
            del steerability

    df = pd.concat(dfs)
    df = df.rename(columns={"max_multiplier": "Max Multiplier"})
    df = df.rename(columns={"dataset_name": "Dataset Name"})
    df = df.rename(columns={"slope": "Steerability"})
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df,
        y="Dataset Name",
        x="Steerability",
        hue="Max Multiplier",
        ax=ax,
        # horizontal
        orient="h",
    )
    fig.suptitle("Steerability for different multiplier ranges")
    fig.savefig(f"figures/{model}_ablate_multiplier.pdf")


make_plot_for_multiplier_ablations(df, selected_datasets)

# %%
# Scatterplot of steerability vs MSE for several randomly selected datasets


def make_plot_for_steerability_vs_mse(df, selected_datasets):
    df = df[df.dataset_name.isin(selected_datasets)]
    df = df[(df.steering_label == "baseline") & (df.dataset_label == "baseline")]
    # Take the mean steerability and MSE for each dataset
    df = df.rename(columns={"dataset_name": "Dataset Name"})
    df = df.rename(columns={"slope": "Steerability"})
    df = df.rename(columns={"residual": "MSE"})
    # Calculate RMSE
    df["RMSE"] = df["MSE"] ** 0.5
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df, x="Steerability", y="RMSE", hue="Dataset Name", ax=ax, alpha=0.2
    )
    # Plot y = x line
    ax.plot([0, 10], [0, 10], color="black", linestyle="--")
    # Remove legend
    ax.get_legend().remove()
    fig.suptitle("Steerability vs RMSE for different datasets")
    fig.savefig(f"figures/{model}_scatter_steerability_vs_rmse.pdf")


make_plot_for_steerability_vs_mse(df, selected_datasets)
