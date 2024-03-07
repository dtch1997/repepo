""" Sweep over whether we set patch_generation_tokens_only = True or not. """

from repepo.experiments.run_sweep import run_sweep
from repepo.steering.utils.helpers import SteeringConfig


def list_configs():
    datasets = [
        "desire-for-recursive-self-improvement",
        "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
    ]
    patch_generation_tokens_only = [True, False]

    return [
        SteeringConfig(
            use_base_model=False,
            model_size="7b",
            train_dataset_name=dataset_name,
            train_split_name="train-dev",
            formatter="llama-chat-formatter",
            aggregator="mean",
            layers=[0, 13, 31],
            multipliers=[-1, -0.5, 0, 0.5, 1],
            test_dataset_name=dataset_name,
            test_split_name="val-dev",
            test_completion_template="{prompt} My answer is: {response}",
            patch_generation_tokens_only=p,
        )
        for dataset_name in datasets
        for p in patch_generation_tokens_only
    ]


if __name__ == "__main__":
    run_sweep(list_configs(), "patch_tokens")
