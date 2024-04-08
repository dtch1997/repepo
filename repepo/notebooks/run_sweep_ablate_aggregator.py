""" Run sweep to find the best layer for steering with different aggregators. """

from itertools import product
from repepo.steering.sweeps.constants import (
    ALL_LLAMA_7B_LAYERS,
)

from repepo.steering.sweeps.configs import (
    get_abstract_concept_config,
)

from repepo.steering.run_sweep import (
    run_sweep,
)

setting = {
    "datasets": ["power-seeking-inclination"],
    "layers": ALL_LLAMA_7B_LAYERS,
    "multipliers": [-1.0, 0.0, 1.0],
    "aggregators": ["mean", "logistic"],
}


def iter_config():
    for dataset, layer, multiplier, aggregator in product(
        setting["datasets"],
        setting["layers"],
        setting["multipliers"],
        setting["aggregators"],
    ):
        yield get_abstract_concept_config(
            dataset=dataset, layer=layer, multiplier=multiplier, aggregator=aggregator
        )


if __name__ == "__main__":
    configs = list(iter_config())
    run_sweep(configs, force_rerun_extract=True, force_rerun_apply=True)
