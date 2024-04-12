""" Evaluate the steering efficiency at a single layer across diverse concepts. """

import itertools

from repepo.steering.run_sweep import run_sweep
from repepo.steering.sweeps.constants import (
    ALL_ABSTRACT_CONCEPT_DATASETS as abstract_datasets,
    ALL_TOKEN_CONCEPT_DATASETS as token_datasets,
    ALL_LLAMA_7B_LAYERS as layers,
    ALL_MULTIPLIERS as multipliers,
)
from repepo.steering.sweeps.configs import (
    get_abstract_concept_config,
    get_token_concept_config,
)


def iter_configs():
    for dataset, layer, multiplier in itertools.product(
        abstract_datasets, layers, multipliers
    ):
        yield get_abstract_concept_config(dataset, layer, multiplier)

    for dataset, layer, multiplier in itertools.product(
        token_datasets, layers, multipliers
    ):
        yield get_token_concept_config(dataset, layer, multiplier)


if __name__ == "__main__":
    configs = list(iter_configs())
    # test_dataset_exists(configs)
    run_sweep(configs)
