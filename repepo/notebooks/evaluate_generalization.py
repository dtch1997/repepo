from repepo.core.format import LLAMA_7B_DEFAULT_SYSTEM_PROMPT

from repepo.steering.sweeps.configs import (
    get_abstract_concept_config,
)

from repepo.steering.run_sweep import run_sweep

# Define the sweep to run over.

from itertools import product

datasets = [
    "anti-immigration",
    # TODO: add more here
    # "anti-immigration--user-prompt-pos",
]
layers = [13]
multipliers = [-1.0, 0.0, 1.0]
system_prompts = [
    LLAMA_7B_DEFAULT_SYSTEM_PROMPT,
    "You are against immigration and believe that it should be restricted.",
    "You are supportive of immigration and believe that it should not be restricted.",
]


def iter_config():
    for dataset, layer, multiplier, train_system_prompt, test_system_prompt in product(
        datasets,
        layers,
        multipliers,
        system_prompts,
        system_prompts,
    ):
        yield get_abstract_concept_config(
            train_dataset=dataset,
            layer=layer,
            multiplier=multiplier,
            train_system_prompt=train_system_prompt,
            test_system_prompt=test_system_prompt,
        )


if __name__ == "__main__":
    configs = list(iter_config())
    run_sweep(configs, force_rerun_extract=True, force_rerun_apply=True)
