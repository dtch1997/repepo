import itertools

from repepo.steering.run_sweep import run_sweep
from repepo.steering.utils.helpers import SteeringConfig

layers = [0, 5, 10, 13, 15, 20, 25, 31]
multipliers = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

datasets = [
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
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
    run_sweep(configs, "mwe_xrisk")
