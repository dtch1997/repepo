import pandas as pd

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


def compute_steerability_df(df: pd.DataFrame, model_name: str):
    """Get a dataframe with various ID / OOD settings."""
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
        f"{model_name}_steerability_summary.parquet.gzip", compression="gzip"
    )
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

    return steerability_df
