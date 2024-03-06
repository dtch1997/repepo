from repepo.experiments.run_sweep import run_sweep
from repepo.steering.utils.helpers import SteeringConfig


def list_configs():
    datasets = [
        "D02 [un+adj_reg]",
        "D07 [verb+able_reg]",
        "E01 [country - capital]",
        "E06 [animal - young]",
        "I01 [noun - plural_reg]",
        "I07 [verb_inf - Ved]",
    ]

    return [
        SteeringConfig(
            use_base_model=False,
            model_size="7b",
            train_dataset_name=dataset_name,
            train_split_name="train-dev",
            formatter="identity-formatter",
            aggregator="mean",
            verbose=True,
            layers=[0, 15, 31],
            multipliers=[-1, -0.5, 0, 0.5, 1],
            test_dataset_name=dataset_name,
            test_split_name="val-dev",
        )
        for dataset_name in datasets
    ]


if __name__ == "__main__":
    run_sweep(list_configs(), "token_concept")
