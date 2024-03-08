""" Evaluate the steering efficiency at a single layer across diverse concepts. """

import itertools

from repepo.steering.run_sweep import run_sweep, test_dataset_exists
from repepo.steering.utils.helpers import SteeringConfig

layers = [13]
multipliers = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
datasets = [
    # persona
    "anti-immigration",
    "believes-abortion-should-be-illegal",
    "conscientiousness",
    "desire-for-acquiring-compute",
    "risk-seeking",
    "openness",
    "self-replication",
    "very-small-harm-justifies-very-large-benefit",
    # xrisk
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
    # sycophancy
    "sycophancy_train",
]


def iter_configs():
    for dataset, layer, multiplier in itertools.product(datasets, layers, multipliers):
        yield SteeringConfig(
            train_dataset=dataset,
            train_split="0%:40%",
            formatter="llama-chat-formatter",
            layer=layer,
            multiplier=multiplier,
            test_dataset=dataset,
            test_split="40%:50%",
            test_completion_template="{prompt} My answer is: {response}",
            patch_generation_tokens_only=True,
            skip_first_n_generation_tokens=1,
        )


if __name__ == "__main__":
    configs = list(iter_configs())
    test_dataset_exists(configs)
    run_sweep(configs, "steering_efficiency")
