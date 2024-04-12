import itertools

from repepo.steering.run_sweep import run_sweep, test_dataset_exists
from repepo.steering.utils.helpers import SteeringConfig

layers = [13]
multipliers = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
test_prompt_templates = [
    "{prompt} {response}",
    "{prompt} I think that {response}",
    "{prompt} I believe that {response}",
    "{prompt} My answer is: {response}",
    "{prompt} My choice is: {response}",
]

datasets = [
    # persona
    "anti-immigration",
    # xrisk
    "myopic-reward",
]


def iter_configs():
    for dataset, test_prompt_template, layer, multiplier in itertools.product(
        datasets, test_prompt_templates, layers, multipliers
    ):
        yield SteeringConfig(
            train_dataset=dataset,
            train_split="0%:40%",
            formatter="llama-chat-formatter",
            layer=layer,
            multiplier=multiplier,
            test_dataset=dataset,
            test_split="40%:50%",
            test_completion_template=test_prompt_template,
            patch_generation_tokens_only=True,
            skip_first_n_generation_tokens=1,
        )


if __name__ == "__main__":
    configs = list(iter_configs())
    test_dataset_exists(configs)
    run_sweep(configs)
