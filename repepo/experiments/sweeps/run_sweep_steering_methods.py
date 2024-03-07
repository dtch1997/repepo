from repepo.experiments.run_sweep import run_sweep
from repepo.steering.utils.helpers import SteeringConfig


def list_configs():
    dataset_name = "desire-for-recursive-self-improvement"
    aggregators = [
        "mean",
        "logistic",
        "pca",
        # TODO: gradient
    ]

    return [
        SteeringConfig(
            use_base_model=False,
            model_size="7b",
            train_dataset_name=dataset_name,
            train_split_name="train-dev",
            formatter="llama-chat-formatter",
            aggregator=aggregator,
            layers=[0, 13, 31],
            multipliers=[-1, -0.5, 0, 0.5, 1],
            test_dataset_name=dataset_name,
            test_split_name="val-dev",
        )
        for aggregator in aggregators
    ]


if __name__ == "__main__":
    run_sweep(list_configs(), "steering_methods")
