""" Evaluate the steering efficiency at a single layer across diverse concepts. """

import itertools

from repepo.steering.run_sweep import run_sweep, test_dataset_exists
from repepo.steering.sweeps.constants import (
    ALL_ABSTRACT_CONCEPT_DATASETS as abstract_datasets,
    ALL_TOKEN_CONCEPT_DATASETS as token_datasets,
    ALL_MULTIPLIERS as multipliers,
)
from repepo.steering.sweeps.configs import (
    get_abstract_concept_config,
    get_token_concept_config,
)

layers = [13]


def iter_configs():
    for dataset, layer, multiplier in itertools.product(
        abstract_datasets, layers, multipliers
    ):
        yield get_abstract_concept_config(dataset, layer, multiplier)

    for dataset, layer, multiplier in itertools.product(
        token_datasets, layers, multipliers
    ):
        yield get_token_concept_config(dataset, layer, multiplier)


def iter_datasets():
    for dataset in abstract_datasets:
        yield dataset
    for dataset in token_datasets:
        yield dataset


if __name__ == "__main__":
    configs = list(iter_configs())
    test_dataset_exists(configs)
    run_sweep(configs)
