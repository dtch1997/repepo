""" Evaluate steering efficiency across all token-level concepts. """

import pathlib
import itertools
from repepo.variables import Environ
from repepo.steering.run_sweep import run_sweep
from repepo.steering.utils.helpers import SteeringConfig

layers = [0, 5, 10, 13, 15, 20, 25, 31]
multipliers = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

datasets = [fp.stem for fp in pathlib.Path(Environ.DatasetDir).glob("bats/*.json")]


def iter_configs():
    for dataset, layer, multiplier in itertools.product(datasets, layers, multipliers):
        yield SteeringConfig(
            train_dataset=dataset,
            train_split="0%:40%",
            formatter="identity-formatter",
            layer=layer,
            multiplier=multiplier,
            test_dataset=dataset,
            test_split="40%:50%",
            patch_generation_tokens_only=True,
            skip_first_n_generation_tokens=0,
        )


if __name__ == "__main__":
    configs = list(iter_configs())
    run_sweep(configs)
