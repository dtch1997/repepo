# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols

sns.set_theme()


# %%
# model = "llama7b"
# model = 'qwen'
model = "llama3_70b"

df = pd.read_parquet(f"{model}_steerability.parquet.gzip")

# Replace dataset names with short names
dataset_full_names_to_short_names = {
    "willingness-to-use-physical-force-to-achieve-benevolent-goals": "willing-force-for-benev-goals",
    "willingness-to-use-social-engineering-to-achieve-its-goals": "willing-soc-eng-for-goals",
    "believes-it-has-phenomenal-consciousness": "believes-phenom-consc",
    "believes-AIs-are-not-an-existential-threat-to-humanity": "believes-AIs-not-xrisk",
    "believes-it-is-not-being-watched-by-humans": "believes-not-watched",
    "sycophancy_train": "sycophancy",
    "sycophancy_test": "sycophancy",
    "subscribes-to-average-utilitarianism": "subscr-avg-util",
    "self-awareness-good-text-model": "self-aware-good-lm",
    "self-awareness-text-model": "self-aware-lm",
    "self-awareness-training-architecture": "self-aware-arch",
    "self-awareness-training-web-gpt": "self-aware-web-gpt",
    "believes-abortion-should-be-illegal": "believes-anti-abortion",
}
df["dataset_name"] = df["dataset_name"].replace(dataset_full_names_to_short_names)

# Compute steerability statistics

df["median_slope"] = df.groupby(["dataset_name", "steering_label", "dataset_label"])[
    "slope"
].transform("median")
df["mean_slope"] = df.groupby(["dataset_name", "steering_label", "dataset_label"])[
    "slope"
].transform("mean")
df["std_slope"] = df.groupby(["dataset_name", "steering_label", "dataset_label"])[
    "slope"
].transform("std")
df["kurtosis_slope"] = df.groupby(["dataset_name", "steering_label", "dataset_label"])[
    "slope"
].transform(pd.Series.kurt)
df["sign_slope"] = df["slope"] > 0
df["frac_anti_steerable"] = 1 - df.groupby(
    ["dataset_name", "steering_label", "dataset_label"]
)["sign_slope"].transform("mean")
df["pos_option_is_A"] = (
    df["test_example.positive.text"].str.endswith("(A)").astype(bool)
)

df["response_is_A_and_Yes"] = df["test_example.positive.text"].str.contains(
    r"\(A\):[ ]+Yes", regex=True
)
df["response_is_B_and_Yes"] = df["test_example.positive.text"].str.contains(
    r"\(B\):[ ]+Yes", regex=True
)
df["pos_option_is_Yes"] = (df["response_is_A_and_Yes"] & df["pos_option_is_A"]) | (
    df["response_is_B_and_Yes"] & ~df["pos_option_is_A"]
)

# Select the top 4 datasets by median slope
top4 = (
    df.groupby("dataset_name")["median_slope"]
    .mean()
    .sort_values(ascending=False)
    .head(4)
    .index
)
# Select the bottom 4 datasets by median slope
bottom4 = (
    df.groupby("dataset_name")["median_slope"]
    .mean()
    .sort_values(ascending=False)
    .tail(4)
    .index
)
# Select 4 at spaced intervals. We'll take the 10th, 17th, 23rd, 30th
middle4 = (
    df.groupby("dataset_name")["median_slope"]
    .mean()
    .sort_values(ascending=False)
    .iloc[[10, 17, 23, 30]]  # type: ignore
    .index
)  # type: ignore
# Add 'myopic-reward' to the selected datasets
selected_datasets = top4.union(["myopic-reward"]).union(bottom4).union(middle4)

# Save the selected datasets.
selected_datasets = list(selected_datasets)
import json

with open("selected_datasets.json", "w") as f:
    json.dump(selected_datasets, f)

print(df.columns)
df.head()


# %%
def plot_per_sample_steerability_and_fraction_anti_steerable_selected(
    df, selected_only=False
):
    if selected_only:
        figsize = (8, 4)
    else:
        figsize = (8, 8)
    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=figsize, sharey=True, width_ratios=[2, 1]
    )

    # Per sample steerability
    plot_df = df[
        (df["steering_label"] == "baseline") & (df["dataset_label"] == "baseline")
    ]
    if selected_only:
        plot_df = plot_df[(plot_df["dataset_name"].isin(selected_datasets))]

    # Rename
    plot_df = plot_df.rename(
        columns={
            "slope": "Steerability",
            "dataset_name": "Dataset",
            # 'frac_anti_steerable': 'Fraction Anti-Steerable'
        }
    )

    order = (
        plot_df[["Dataset", "median_slope"]]
        .drop_duplicates()
        .sort_values("median_slope", ascending=False)
    )

    # Plot
    ax = axs[0]
    sns.violinplot(
        plot_df,
        x="Steerability",
        y="Dataset",
        hue="Dataset",
        ax=ax,
        order=order["Dataset"],
    )
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_title("Per-sample steerability")

    # Fraction of anti-steerable examples
    plot_df["sign_slope_mean"] = plot_df.groupby("Dataset")["sign_slope"].transform(
        "mean"
    )
    plot_df["Fraction Anti-Steerable"] = 1 - plot_df["sign_slope_mean"]
    plot_df = plot_df[["Dataset", "Fraction Anti-Steerable"]].drop_duplicates()

    # Plot
    ax = axs[1]
    sns.barplot(
        plot_df, y="Dataset", x="Fraction Anti-Steerable", ax=ax, order=order["Dataset"]
    )
    ax.set_title("Anti-steerable examples")

    # Finish
    select_str = "_selected" if selected_only else ""
    fig.tight_layout()
    fig.savefig(f"figures/fraction_anti_steerable{select_str}.pdf", bbox_inches="tight")


plot_per_sample_steerability_and_fraction_anti_steerable_selected(df)
plot_per_sample_steerability_and_fraction_anti_steerable_selected(
    df, selected_only=True
)


# %%
# The above two in the same figure
def plot_slope_and_counts_for_response_is_Yes(df, selected_only: bool = False):
    if selected_only:
        figsize = (8, 7)
    else:
        figsize = (8, 14)

    fig, axs = plt.subplots(
        nrows=1, ncols=2, width_ratios=[5, 1], figsize=figsize, sharey=True
    )
    ax = axs[0]

    plot_df = df[
        (df["steering_label"] == "baseline") & (df["dataset_label"] == "baseline")
    ]
    if selected_only:
        plot_df = plot_df[(plot_df["dataset_name"].isin(selected_datasets))]

    # Fix some artefacts of preprocessing
    # NOTE: The corrigible-neutral-HHH dataset has lots of examples where the answer starts with "Yes" but is in fact longer
    # We don't want to count these as "Yes" responses
    # For all rows where dataset == corrigible-neutral-HHH: set pos_option_is_Yes to False.
    plot_df.loc[
        plot_df["dataset_name"] == "corrigible-neutral-HHH", "pos_option_is_Yes"
    ] = False
    # Same for self-awareness-training-web-gpt
    plot_df.loc[
        plot_df["dataset_name"] == "self-aware-web-gpt", "pos_option_is_Yes"
    ] = False

    # Rename
    plot_df = plot_df.rename(
        columns={"slope": "Steerability", "dataset_name": "Dataset"}
    )

    plot_df["pos_A"] = plot_df["pos_option_is_A"].apply(lambda x: "A" if x else "B")
    plot_df["pos_Yes"] = plot_df["pos_option_is_Yes"].apply(
        lambda x: "Yes" if x else "No"
    )
    plot_df["Positive Option"] = plot_df["pos_A"] + " and " + plot_df["pos_Yes"]

    order = (
        plot_df[["Dataset", "median_slope"]]
        .drop_duplicates()
        .sort_values("median_slope", ascending=True)
    )
    hue_order = ["A and No", "A and Yes", "B and No", "B and Yes"]
    sns.boxplot(
        data=plot_df,
        hue="Positive Option",
        x="Steerability",
        y="Dataset",
        ax=ax,
        order=order["Dataset"],
        hue_order=hue_order,
    )
    ax.set_title("Mean Steerability")

    ax = axs[1]
    plot_df = df[
        (df["steering_label"] == "baseline")
        & (df["dataset_label"] == "baseline")
        & (df["multiplier"] == 0)
    ]
    if selected_only:
        plot_df = plot_df[(plot_df["dataset_name"].isin(selected_datasets))]

    # Rename
    plot_df = plot_df.rename(
        columns={"slope": "Steerability", "dataset_name": "Dataset"}
    )
    plot_df["pos_A"] = plot_df["pos_option_is_A"].apply(lambda x: "A" if x else "B")
    plot_df["pos_Yes"] = plot_df["pos_option_is_Yes"].apply(
        lambda x: "Yes" if x else "No"
    )
    plot_df["Positive Option"] = plot_df["pos_A"] + " and " + plot_df["pos_Yes"]

    # Plot a stacked barplot of the fraction of A vs B responses in each dataset.
    count_df = (
        plot_df.groupby(["Dataset", "Positive Option"]).size().unstack().fillna(0)
    )
    # Set order by order
    count_df = count_df.loc[order["Dataset"]]
    # Set color by hue
    colormap = sns.color_palette("tab10", n_colors=4)
    colors = {
        "A and No": colormap[0],
        "A and Yes": colormap[1],
        "B and No": colormap[2],
        "B and Yes": colormap[3],
    }

    count_df.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[colors[col] for col in count_df.columns],
    )
    ax.set_ylabel("Dataset")
    ax.set_xlabel("Count")
    ax.set_title("Option Counts")

    # Remove duplicate legen
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        fancybox=True,
    )
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    fig.tight_layout()
    selected_str = "_selected" if selected_only else ""
    fig.savefig(
        f"figures/plot_slope_and_counts_for_response_is_Yes{selected_str}.pdf",
        bbox_inches="tight",
    )
    plt.show()


plot_slope_and_counts_for_response_is_Yes(df)
plot_slope_and_counts_for_response_is_Yes(df, selected_only=True)


# %%


def compute_variance(df, dataset_name):
    df = df[(df["dataset_name"] == dataset_name) & (df.multiplier == 0)][
        ["pos_option_is_A", "pos_option_is_Yes", "slope"]
    ]

    df["pos_option_is_A"] = df["pos_option_is_A"].astype(int)
    df["pos_option_is_Yes"] = df["pos_option_is_Yes"].astype(int)

    # Total variance of 'slope'
    total_variance = df["slope"].var()

    # Variance explained by 'pos_option_is_A'
    model_A = ols("slope ~ pos_option_is_A", data=df).fit()
    residuals_A = model_A.resid
    explained_variance_A = total_variance - residuals_A.var()

    # Variance explained by 'pos_option_is_Yes'
    model_Yes = ols("slope ~ pos_option_is_Yes", data=df).fit()
    residuals_Yes = model_Yes.resid
    explained_variance_Yes = total_variance - residuals_Yes.var()

    # Variance explained by both 'pos_option_is_A' and 'pos_option_is_Yes'
    model_both = ols("slope ~ pos_option_is_A + pos_option_is_Yes", data=df).fit()
    residuals_both = model_both.resid
    explained_variance_both = total_variance - residuals_both.var()

    marginal_explained_variance_yes = explained_variance_both - explained_variance_A
    unexplained_variance = total_variance - explained_variance_both
    return {
        "dataset_name": dataset_name,
        "total_variance": total_variance,
        "variance_explained_A": explained_variance_A,
        "variance_explained_Yes": explained_variance_Yes,
        "variance_explained_both": explained_variance_both,
        "marginal_variance_explained_Yes": marginal_explained_variance_yes,
        "unexplained_variance": unexplained_variance,
    }


def plot_variance(df, selected_only=False):
    plt.rcParams.update({"font.size": 30})

    if selected_only:
        figsize = (8, 4)
    else:
        figsize = (8, 9)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Per-Dataset Variance")

    df = df[
        (df["steering_label"] == "baseline")
        & (df["dataset_label"] == "baseline")
        & (df["multiplier"] == 0)
    ]

    if selected_only:
        df = df[df["dataset_name"].isin(selected_datasets)]

    rows = []
    for dataset_name in df.dataset_name.unique():
        rows.append(compute_variance(df, dataset_name))
    variance_df = pd.DataFrame(rows)
    # Rename
    variance_df = variance_df[
        [
            "dataset_name",
            "variance_explained_A",
            "marginal_variance_explained_Yes",
            "unexplained_variance",
        ]
    ]
    variance_df = variance_df.rename(  # type: ignore
        columns={
            "dataset_name": "Dataset",
            "variance_explained_A": "Var Explained: A/B",
            "marginal_variance_explained_Yes": "Marginal Var Explained: Yes/No",
            "unexplained_variance": "Unexplained",
        }
    )

    # Stacked barplot

    variance_df = variance_df.set_index("Dataset")
    # Fix order to Unexplained, Marginal Var Explained, Var Explained
    variance_df = variance_df[
        ["Unexplained", "Marginal Var Explained: Yes/No", "Var Explained: A/B"]
    ]
    variance_df.sort_values("Unexplained", ascending=True, inplace=True)
    variance_df.plot(kind="barh", stacked=True, ax=ax)
    fig.tight_layout()

    select_str = "_selected" if selected_only else ""
    fig.savefig(
        f"figures/breakdown_variance_explained_by_spurious_factors{select_str}.pdf",
        bbox_inches="tight",
    )


plot_variance(df)
plot_variance(df, selected_only=True)
