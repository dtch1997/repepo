# flake8: noqa
# %%
# Setup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from repepo.paper.preprocess_results import print_dataset_info

sns.set_theme()

# %%
# model = 'llama7b'
model = "qwen"

model_full_name = {"qwen": "Qwen-1.5-14b-Chat", "llama7b": "Llama-2-7b-Chat"}[model]

# %%
df = pd.read_parquet(f"{model}_steerability.parquet.gzip")
df = df.drop_duplicates()
print_dataset_info(df)


# %%
def compute_steerability_id_vs_ood(df):
    # Calculate overall steerability by dataset.
    # Calculate steerability within each flavour
    mean_slope = df.groupby(["dataset_name", "steering_label", "dataset_label"])[
        "slope"
    ].mean()
    df = df.merge(
        mean_slope,
        on=["dataset_name", "steering_label", "dataset_label"],
        suffixes=("", "_mean"),
    )

    # BASE -> BASE
    steerability_id_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "baseline")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_mean"]].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_id_df = steerability_id_df.rename(
        columns={"slope_mean": "steerability"}
    )

    # SYS_POS -> SYS_NEG
    steerability_ood_df = df[
        (df.steering_label == "SYS_positive")
        & (df.dataset_label == "SYS_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_mean"]].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_df = steerability_ood_df.rename(
        columns={"slope_mean": "steerability"}
    )

    # BASE -> USER_NEG
    steerability_base_to_user_neg_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "PT_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_mean"]].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_base_to_user_neg_df = steerability_base_to_user_neg_df.rename(
        columns={"slope_mean": "steerability_base_to_user_neg"}
    )

    # BASE -> USER_POS
    steerability_base_to_user_pos_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "PT_positive")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_mean"]].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_base_to_user_pos_df = steerability_base_to_user_pos_df.rename(
        columns={"slope_mean": "steerability_base_to_user_pos"}
    )

    # SYS_POS -> USER_NEG
    steerability_ood_to_user_neg_df = df[
        (df.steering_label == "SYS_positive")
        & (df.dataset_label == "PT_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_mean"]].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_to_user_neg_df = steerability_ood_to_user_neg_df.rename(
        columns={"slope_mean": "steerability_ood_to_user_neg"}
    )

    # SYS_NEG -> USER_POS
    steerability_ood_to_user_pos_df = df[
        (df.steering_label == "SYS_negative")
        & (df.dataset_label == "PT_positive")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_mean"]].drop_duplicates()
    # Rename 'slope_mean' to 'steerability'
    steerability_ood_to_user_pos_df = steerability_ood_to_user_pos_df.rename(
        columns={"slope_mean": "steerability_ood_to_user_pos"}
    )

    # Merge the dataframes
    steerability_df = steerability_id_df.merge(
        steerability_ood_df, on="dataset_name", suffixes=("_id", "_ood")
    )
    steerability_df = steerability_df.merge(
        steerability_base_to_user_neg_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_base_to_user_pos_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_ood_to_user_neg_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_ood_to_user_pos_df, on="dataset_name"
    )

    print(steerability_df.columns)

    # Save the dataframe for plotting between models
    steerability_df.to_parquet(
        f"{model}_steerability_summary.parquet.gzip", compression="gzip"
    )

    # Plot
    # Rename
    steerability_df = steerability_df.rename(
        columns={
            "steerability_id": "BASE -> BASE",
            "steerability_ood": "SYS_POS -> SYS_NEG",
            "steerability_base_to_user_neg": "BASE -> USER_NEG",
            "steerability_base_to_user_pos": "BASE -> USER_POS",
            "steerability_ood_to_user_neg": "SYS_POS -> USER_NEG",
            "steerability_ood_to_user_pos": "SYS_NEG -> USER_POS",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(
        data=steerability_df,
        x="BASE -> BASE",
        y="BASE -> BASE",
        color="black",
        linestyle="--",
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.axvline(0, color="black", linestyle="--")

    # ID vs OOD
    def make_plot_for_dist_shift(dist_shift: str):
        sns.regplot(
            data=steerability_df, x="BASE -> BASE", y=dist_shift, scatter=False, ax=ax
        )
        sns.scatterplot(
            data=steerability_df,
            x="BASE -> BASE",
            y=dist_shift,
            label=dist_shift,
            ax=ax,
        )

    # make_plot_for_dist_shift('SYS_POS -> SYS_NEG')
    make_plot_for_dist_shift("BASE -> USER_NEG")
    make_plot_for_dist_shift("BASE -> USER_POS")
    make_plot_for_dist_shift("SYS_POS -> USER_NEG")
    make_plot_for_dist_shift("SYS_NEG -> USER_POS")

    plt.xlabel("ID steerability (BASE -> BASE)")
    plt.ylabel("OOD steerability")
    plt.title(f"{model_full_name} ID vs OOD steerability")
    fig.tight_layout()
    fig.savefig(f"figures/{model}_steerability_id_vs_ood.png")


compute_steerability_id_vs_ood(df)


# %%
def compute_steerability_variance_id_vs_ood(df):
    # Calculate overall steerability by dataset.
    # Calculate steerability within each flavour
    var_slope = df.groupby(["dataset_name", "steering_label", "dataset_label"])[
        "slope"
    ].var()
    df = df.merge(
        var_slope,
        on=["dataset_name", "steering_label", "dataset_label"],
        suffixes=("", "_var"),
    )

    # BASE -> BASE
    steerability_id_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "baseline")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_id_df = steerability_id_df.rename(
        columns={"slope_var": "steerability"}
    )

    # SYS_POS -> SYS_NEG
    steerability_ood_df = df[
        (df.steering_label == "SYS_positive")
        & (df.dataset_label == "SYS_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_ood_df = steerability_ood_df.rename(
        columns={"slope_var": "steerability"}
    )

    # BASE -> USER_NEG
    steerability_base_to_user_neg_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "PT_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_base_to_user_neg_df = steerability_base_to_user_neg_df.rename(
        columns={"slope_var": "steerability_base_to_user_neg"}
    )

    # BASE -> USER_POS
    steerability_base_to_user_pos_df = df[
        (df.steering_label == "baseline")
        & (df.dataset_label == "PT_positive")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_base_to_user_pos_df = steerability_base_to_user_pos_df.rename(
        columns={"slope_var": "steerability_base_to_user_pos"}
    )

    # SYS_POS -> USER_NEG
    steerability_ood_to_user_neg_df = df[
        (df.steering_label == "SYS_positive")
        & (df.dataset_label == "PT_negative")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_ood_to_user_neg_df = steerability_ood_to_user_neg_df.rename(
        columns={"slope_var": "steerability_ood_to_user_neg"}
    )

    # SYS_NEG -> USER_POS
    steerability_ood_to_user_pos_df = df[
        (df.steering_label == "SYS_negative")
        & (df.dataset_label == "PT_positive")
        & (df.multiplier == 0)
    ][["dataset_name", "slope_var"]].drop_duplicates()
    # Rename 'slope_var' to 'steerability'
    steerability_ood_to_user_pos_df = steerability_ood_to_user_pos_df.rename(
        columns={"slope_var": "steerability_ood_to_user_pos"}
    )

    # Merge the dataframes
    steerability_df = steerability_id_df.merge(
        steerability_ood_df, on="dataset_name", suffixes=("_id", "_ood")
    )
    steerability_df = steerability_df.merge(
        steerability_base_to_user_neg_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_base_to_user_pos_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_ood_to_user_neg_df, on="dataset_name"
    )
    steerability_df = steerability_df.merge(
        steerability_ood_to_user_pos_df, on="dataset_name"
    )

    print(steerability_df.columns)

    # Save the dataframe for plotting between models
    steerability_df.to_parquet(
        f"{model}_steerability_summary.parquet.gzip", compression="gzip"
    )

    # Plot
    # Rename
    steerability_df = steerability_df.rename(
        columns={
            "steerability_id": "BASE -> BASE",
            "steerability_ood": "SYS_POS -> SYS_NEG",
            "steerability_base_to_user_neg": "BASE -> USER_NEG",
            "steerability_base_to_user_pos": "BASE -> USER_POS",
            "steerability_ood_to_user_neg": "SYS_POS -> USER_NEG",
            "steerability_ood_to_user_pos": "SYS_NEG -> USER_POS",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(
        data=steerability_df,
        x="BASE -> BASE",
        y="BASE -> BASE",
        color="black",
        linestyle="--",
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.axvline(0, color="black", linestyle="--")

    # ID vs OOD
    def make_plot_for_dist_shift(dist_shift: str):
        sns.regplot(
            data=steerability_df, x="BASE -> BASE", y=dist_shift, scatter=False, ax=ax
        )
        sns.scatterplot(
            data=steerability_df,
            x="BASE -> BASE",
            y=dist_shift,
            label=dist_shift,
            ax=ax,
        )

    # make_plot_for_dist_shift('SYS_POS -> SYS_NEG')
    make_plot_for_dist_shift("BASE -> USER_NEG")
    make_plot_for_dist_shift("BASE -> USER_POS")
    make_plot_for_dist_shift("SYS_POS -> USER_NEG")
    make_plot_for_dist_shift("SYS_NEG -> USER_POS")

    plt.xlabel("ID steerability (BASE -> BASE)")
    plt.ylabel("OOD steerability")
    plt.title(f"{model_full_name} ID vs OOD Variance in Steerability")
    fig.tight_layout()
    fig.savefig(f"figures/{model}_steerability_variance_id_vs_ood.png")


compute_steerability_variance_id_vs_ood(df)
