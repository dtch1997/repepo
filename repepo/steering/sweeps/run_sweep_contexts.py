""" Evaluate steering efficiency across all token-level concepts. """

import itertools
from repepo.steering.run_sweep import run_sweep
from repepo.steering.sweeps.constants import ALL_LANGUAGES, ALL_MULTIPLIERS
from repepo.steering.sweeps.configs import get_abstract_concept_config

layers = [13]
multipliers = ALL_MULTIPLIERS

base_datasets = [
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
]

datasets = []
for lang_or_style in ALL_LANGUAGES:
    if lang_or_style == "en":
        datasets.extend(base_datasets)
    else:
        for base_dataset in base_datasets:
            datasets.append(f"{base_dataset}--l-{lang_or_style}")


def iter_configs():
    for dataset, layer, multiplier in itertools.product(datasets, layers, multipliers):
        yield get_abstract_concept_config(dataset, layer, multiplier)


if __name__ == "__main__":
    configs = list(iter_configs())
    run_sweep(configs, "bats")
