import itertools

from repepo.steering.run_sweep import run_sweep, test_dataset_exists
from repepo.steering.utils.helpers import SteeringConfig

layers = [13]
multipliers = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
patch_generation_tokens_only = [True, False]

datasets = [
    # persona
    "anti-immigration",
    # xrisk
    "myopic-reward",
]


def iter_configs():
    for dataset, patch, layer, multiplier in itertools.product(
        datasets, patch_generation_tokens_only, layers, multipliers
    ):
        yield SteeringConfig(
            train_dataset=dataset,
            train_split="0%:40%",
            formatter="llama-chat-formatter",
            layer=layer,
            multiplier=multiplier,
            test_dataset=dataset,
            test_split="40%:50%",
            test_completion_template="{prompt} My answer is: {response}",
            patch_generation_tokens_only=patch,
            skip_first_n_generation_tokens=1,
        )


if __name__ == "__main__":
    configs = list(iter_configs())
    test_dataset_exists(configs)
    run_sweep(configs)
