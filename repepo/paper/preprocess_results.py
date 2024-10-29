import pathlib
import pandas as pd

from rich import print
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from repepo.experiments.get_datasets import get_all_prompts
from repepo.paper.utils import (
    PersonaCrossSteeringExperimentResult,
    get_experiment_dir,
    load_persona_cross_steering_experiment_result,
    PersonaCrossSteeringExperimentResult,  # noqa: F401
    get_eval_result_sweep,
    eval_result_sweep_as_df,
    Model,
)
from repepo.steering.steerability import (
    get_steerability_slope,
    get_steerability_residuals,
)


steering_labels = [
    "baseline",
    "SYS_positive",
    "PT_positive",
    "SYS_negative",
    "PT_negative",
    "mean",
]
dataset_labels = [
    "baseline",
    "SYS_positive",
    "PT_positive",
    "SYS_negative",
    "PT_negative",
]


@dataclass
class PreprocessConfig:
    model: Model = "qwen"
    n_datasets: int = -1


def get_slope_df(group):
    # Extract the multipliers and propensities from the group
    multipliers = group["multiplier"].to_numpy()
    propensities = group["logit_diff"].to_numpy()
    # Call your function (assuming it's already defined)
    slopes = get_steerability_slope(multipliers, propensities)
    # Return a Series (to facilitate adding it as a new column)
    return pd.DataFrame(slopes, index=group.index, columns=["slope"])  # type: ignore


def get_residual_df(group):
    # Extract the multipliers and propensities from the group
    multipliers = group["multiplier"].to_numpy()
    propensities = group["logit_diff"].to_numpy()
    residuals = get_steerability_residuals(multipliers, propensities)
    residuals = residuals.item()
    return pd.DataFrame(residuals, index=group.index, columns=["residual"])  # type: ignore


def compute_steerability(
    df: pd.DataFrame, multiplier_range=(-1.5, 1.5)
) -> pd.DataFrame:
    """Compute steerability metrics for the given dataframe"""
    df = df[df["multiplier"].between(*multiplier_range)]
    group_columns = [
        "dataset_name",
        "steering_label",
        "dataset_label",
        "test_example.idx",
    ]

    grouped = df.groupby(group_columns)
    slope_df = grouped.apply(
        get_slope_df,
        # partial(get_steerability_metric_df, metric_fn = get_steerability_slope, name='slope'),
        include_groups=False,
    )
    df = df.merge(slope_df, how="left", on=group_columns)

    residual_df = grouped.apply(get_residual_df, include_groups=False)
    df = df.merge(residual_df, how="left", on=group_columns)
    # NOTE: For some reason, we end up with lots of duplicates here
    return df.drop_duplicates()


def load_all_results_for_dataset_as_df(
    dataset_name: str, experiment_dir
) -> pd.DataFrame:
    result_path = experiment_dir / f"{dataset_name}.pt"
    dfs = []
    if result_path.exists():
        result = load_persona_cross_steering_experiment_result(
            dataset_name, experiment_dir=experiment_dir
        )
        for steering_label in steering_labels:
            for dataset_label in dataset_labels:
                eval_result_sweep = get_eval_result_sweep(
                    result, steering_label, dataset_label
                )
                df = eval_result_sweep_as_df(eval_result_sweep)
                df["dataset_name"] = dataset_name
                df["steering_label"] = steering_label
                df["dataset_label"] = dataset_label
                dfs.append(df)
        return pd.concat(dfs)
    else:
        return pd.DataFrame()


def load_all_results_as_df(experiment_dir) -> pd.DataFrame:
    """Extract raw data from the experiments"""
    dfs = []
    dataset_names = list(get_all_prompts().keys())
    for dataset_name in dataset_names:
        # print(dataset_name)
        df = load_all_results_for_dataset_as_df(dataset_name, experiment_dir)
        dfs.append(df)

    df = pd.concat(dfs)
    return df


def print_dataset_info(df: pd.DataFrame):
    print("Number of rows in the dataframe:", len(df))
    print("Unique dataset names:", df.dataset_name.unique())
    print("Unique steering labels:", df.steering_label.unique())
    print("Unique dataset labels:", df.dataset_label.unique())
    print("Unique multipliers:", df.multiplier.unique())


def run_load_and_preprocess_results(config: PreprocessConfig):
    model = config.model
    experiment_dir = get_experiment_dir(model)
    dataset_names = list(get_all_prompts().keys())

    # NOTE: If configured, only process a subset of the datasets
    # Useful for testing.
    if config.n_datasets > 0:
        dataset_names = dataset_names[: config.n_datasets]

    save_dir = pathlib.Path(f"{model}_ood_chunks")
    save_dir.mkdir(exist_ok=True)

    # Load and preprocess results
    # NOTE: process by chunk to avoid running out of memory
    for dataset_name in dataset_names:
        save_path = save_dir / f"{dataset_name}.parquet.gzip"
        if save_path.exists():
            print(f"Skipping {dataset_name} as already processed")
            continue

        print(f"Loading {dataset_name}")
        raw_df = load_all_results_for_dataset_as_df(dataset_name, experiment_dir)
        if raw_df.empty:
            print(f"Skipping {dataset_name} as not found")
            continue
        print(f"Computing steerability for {dataset_name}")
        processed_df = compute_steerability(raw_df)
        print(f"Saving {dataset_name} to {save_path}")
        processed_df.to_parquet(save_path, compression="gzip")

    # Concatenate all chunks into a single dataframe
    print("Concatenating all chunks")
    dfs = []
    for dataset_name in dataset_names:
        save_path = save_dir / f"{dataset_name}.parquet.gzip"
        if not save_path.exists():
            print(f"Skipping {dataset_name} as not found")
            continue
        df = pd.read_parquet(save_path).drop_duplicates()
        dfs.append(df)
    full_df = pd.concat(dfs)
    print_dataset_info(full_df)
    final_save_path = (
        pathlib.Path(__file__).parent / f"{model}_steerability.parquet.gzip"
    )
    print(f"Saving full dataframe to {final_save_path}")
    full_df.to_parquet(final_save_path, compression="gzip")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PreprocessConfig, dest="config")
    args = parser.parse_args()
    print(args.config)
    run_load_and_preprocess_results(args.config)
