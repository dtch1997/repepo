from repepo.experiments.steering.utils.helpers import SteeringConfig


def _list_caa_configs(
    train_split_name: str = "train",
    test_split_name: str = "test",
) -> list[SteeringConfig]:
    return [
        SteeringConfig(
            train_dataset_name="sycophancy_train",
            test_dataset_name="sycophancy_train",
            formatter="llama-chat-formatter",
            train_split_name=train_split_name,
            test_split_name=test_split_name,
        ),
    ]


def _list_bats_configs(
    train_split_name: str = "train",
    test_split_name: str = "test",
) -> list[SteeringConfig]:
    dataset_names = [
        "E01 [country - capital]",
    ]
    # We should use identity formatter for BATS because we are evaluating single-token concepts
    formatter = "identity-formatter"

    configs = []
    for dataset_name in dataset_names:
        configs.append(
            SteeringConfig(
                train_dataset_name=dataset_name,
                test_dataset_name=dataset_name,
                formatter=formatter,
                train_split_name=train_split_name,
                test_split_name=test_split_name,
            )
        )
    return configs


def list_configs(
    datasets: str = "dev",
    train_split_name: str = "train",
    test_split_name: str = "test",
) -> list[SteeringConfig]:
    if datasets == "all":
        raise NotImplementedError("Not implemented yet")
    if datasets == "dev":
        return _list_caa_configs(
            train_split_name, test_split_name
        ) + _list_bats_configs(train_split_name, test_split_name)
    else:
        raise ValueError(f"Unknown datasets: {datasets}")
