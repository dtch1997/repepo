""" Evaluate the steering efficiency at a single layer across diverse concepts. """

import itertools

from repepo.steering.run_sweep import run_sweep, test_dataset_exists
from repepo.steering.sweeps.constants import (
    ALL_MULTIPLIERS as multipliers,
    ALL_LLAMA_7B_LAYERS as layers,
)
from repepo.steering.utils.helpers import SteeringConfig

datasets = ["power-seeking-inclination"]
aggregators = ["mean", "logistic"]


def iter_configs():
    for dataset, layer, multiplier, aggregator in itertools.product(
        datasets, layers, multipliers, aggregators
    ):
        yield SteeringConfig(
            train_dataset=dataset,
            train_split="0%:40%",
            formatter="llama-chat-formatter",
            aggregator=aggregator,
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
    run_sweep(configs)
