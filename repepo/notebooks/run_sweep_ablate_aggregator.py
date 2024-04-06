""" Attempts to find the best layer for logistic regression steering. """

from repepo.steering.sweeps.constants import (
    ALL_LLAMA_7B_LAYERS,
)

from repepo.steering.sweeps.configs import (
    get_abstract_concept_config,
)


# Define the sweep to run over.

from itertools import product

debug_setting = {
    "datasets": ["power-seeking-inclination"],
    "layers": ALL_LLAMA_7B_LAYERS,
    "multipliers": [-1.0, 0.0, 1.0],
    "aggregators": ["mean", "logistic"],
}


def iter_config(setting):
    for dataset, layer, multiplier, aggregator in product(
        setting["datasets"],
        setting["layers"],
        setting["multipliers"],
        setting["aggregators"],
    ):
        yield get_abstract_concept_config(
            dataset=dataset, layer=layer, multiplier=multiplier, aggregator=aggregator
        )
