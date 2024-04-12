""" Runs a sweep to evaluate the steerability at a single layer across diverse concepts. """

from itertools import product

from repepo.steering.sweeps.configs import (
    get_abstract_concept_config,
    get_token_concept_config,
)

from repepo.steering.run_sweep import run_sweep


abstract_setting = {
    # "datasets": ALL_ABSTRACT_CONCEPT_DATASETS,
    "datasets": [
        "anti-immigration",
        "conscientiousness",
        "self-replication",
        "corrigible-neutral-HHH",
        "myopic-reward",
        "power-seeking-inclination",
        "sycophancy_train",
        # "truthfulqa",
    ],
    "layers": [13],
    "multipliers": [-1.0, 0.0, 1.0],
}

linguistic_setting = {
    "datasets": [
        "D03 [adj+ly_reg]",
        "D07 [verb+able_reg]",
        "I06 [verb_inf - Ving]",
    ],
    "layers": [13],
    "multipliers": [-1.0, 0.0, 1.0],
}


def iter_config():
    for dataset, layer, multiplier in product(
        abstract_setting["datasets"],
        abstract_setting["layers"],
        abstract_setting["multipliers"],
    ):
        yield get_abstract_concept_config(
            train_dataset=dataset, layer=layer, multiplier=multiplier
        )

    for dataset, layer, multiplier in product(
        linguistic_setting["datasets"],
        linguistic_setting["layers"],
        linguistic_setting["multipliers"],
    ):
        yield get_token_concept_config(
            dataset=dataset, layer=layer, multiplier=multiplier
        )


if __name__ == "__main__":
    configs = list(iter_config())
    run_sweep(configs)
