# %%
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
from sklearn.calibration import column_or_1d
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
    fig.tight_layout()
    fig.show()
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

# %%
# Load the steering results for the selected datasets
import pandas as pd

from repepo.paper.helpers import compute_steerability_df
from repepo.paper.utils import get_model_full_name

llama_df = pd.read_parquet("llama7b_steerability.parquet.gzip").drop_duplicates()
llama_steerability_df = compute_steerability_df(llama_df, "llama7b")

qwen_df = pd.read_parquet("qwen_steerability.parquet.gzip").drop_duplicates()
qwen_steerability_df = compute_steerability_df(qwen_df, "qwen")

gemma_df = pd.read_parquet("gemma_steerability.parquet.gzip").drop_duplicates()
gemma_steerability_df = compute_steerability_df(gemma_df, "gemma")

# %%
print(len(gemma_steerability_df))
print(gemma_steerability_df.columns)
print(qwen_steerability_df.columns)
print(llama_steerability_df.columns)

# %%
# Merge the dataframes


def make_cross_model_df(
    llama_steerability_df,
    qwen_steerability_df,
    gemma_steerability_df,
    select_columns,
):
    steerability_df = (
        llama_steerability_df[select_columns]
        .merge(
            qwen_steerability_df[select_columns],
            on="dataset_name",
            suffixes=("_llama", "_qwen"),
        )
        .merge(
            gemma_steerability_df[select_columns],
            on="dataset_name",
            suffixes=("", "_gemma"),
        )
    )
    # TODO: Why does the gemma name not get updated correctly?
    # NOTE: Manually update
    steerability_df = steerability_df.rename(
        columns={
            "BASE -> BASE": "BASE -> BASE_gemma",
            "SYS_POS -> USER_NEG": "SYS_POS -> USER_NEG_gemma",
        }
    )
    return steerability_df


id_df = make_cross_model_df(
    llama_steerability_df,
    qwen_steerability_df,
    gemma_steerability_df,
    select_columns=["dataset_name", "BASE -> BASE"],
)
ood_df = make_cross_model_df(
    llama_steerability_df,
    qwen_steerability_df,
    gemma_steerability_df,
    select_columns=["dataset_name", "SYS_POS -> USER_NEG"],
)

print(id_df.columns)
print(ood_df.columns)

# %%
# Scatterplot matrix of steerability between different models, both ID and OOD
import matplotlib.pyplot as plt
import seaborn as sns
from repepo.paper.utils import get_model_full_name
from scipy.stats import spearmanr

sns.set_theme()


def add_xy_line(xdata, ydata, xy_min: float, xy_max: float, **kwargs):
    # Add the diagonal line
    plt.xlim(xy_min, xy_max)
    plt.ylim(xy_min, xy_max)
    plt.axline(
        (xy_min, xy_min), (xy_max, xy_max), color="black", linestyle="--", linewidth=2
    )


def add_textbox(xdata, ydata, xy_min: float, xy_max: float, **kwargs):

    spearman_corr, spearman_p = spearmanr(
        xdata, ydata, nan_policy="omit"
    )  # Spearman correlation
    plt.text(0, xy_max - 0.4, f"Corr: {spearman_corr:.2f}", fontsize=12)


def make_scatterplot_matrix(df, type: str, title):
    columns = [
        get_model_full_name("llama7b"),
        get_model_full_name("qwen"),
        get_model_full_name("gemma"),
    ]
    # Rename columns
    df = df.rename(
        columns={
            # ID
            "BASE -> BASE_llama": get_model_full_name("llama7b"),
            "BASE -> BASE_qwen": get_model_full_name("qwen"),
            "BASE -> BASE_gemma": get_model_full_name("gemma"),
            # OOD
            "SYS_POS -> USER_NEG_llama": get_model_full_name("llama7b"),
            "SYS_POS -> USER_NEG_qwen": get_model_full_name("qwen"),
            "SYS_POS -> USER_NEG_gemma": get_model_full_name("gemma"),
        }
    )

    grid = sns.pairplot(df, vars=columns)
    grid.map_offdiag(add_xy_line, xy_min=-2.0, xy_max=6.0)
    fig = grid.figure
    fig.suptitle(title)
    fig.tight_layout()
    grid.map_offdiag(add_textbox, xy_min=-2.0, xy_max=6.0)
    fig.show()
    fig.savefig(f"figures/all_models_scatterplot_matrix_{type}.pdf")


make_scatterplot_matrix(id_df, "id", "Steerability between different models (ID)")
make_scatterplot_matrix(ood_df, "ood", "Steerability between different models (OOD)")
