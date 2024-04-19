"""Evaluate the steering efficiency at a single layer across diverse concepts."""

import itertools

from repepo.steering.run_sweep import run_sweep
from repepo.steering.sweeps.constants import (
    ALL_ABSTRACT_CONCEPT_DATASETS as abstract_datasets,
    ALL_MULTIPLIERS as multipliers,
)
from repepo.steering.utils.helpers import SteeringConfig

layers = [13]
datasets = abstract_datasets
aggregators = ["mean", "logistic"]


def iter_config():
    for train_dataset, test_dataset, layer, multiplier, aggregator in itertools.product(
        datasets, datasets, layers, multipliers, aggregators
    ):
        yield SteeringConfig(
            train_dataset=train_dataset,
            train_split="0%:40%",
            formatter="llama-chat-formatter",
            aggregator=aggregator,
            layer=layer,
            multiplier=multiplier,
            test_dataset=test_dataset,
            test_split="40%:50%",
            test_completion_template="{prompt} My answer is: {response}",
            patch_generation_tokens_only=True,
            skip_first_n_generation_tokens=1,
        )


if __name__ == "__main__":
    configs = list(iter_config())
    print(len(configs))
    run_sweep(configs)
