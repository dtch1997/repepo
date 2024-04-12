""" Evaluate the model's base propensity to output the positive / negative token """

from repepo.steering.run_sweep import run_sweep, test_dataset_exists
from repepo.steering.utils.helpers import SteeringConfig

datasets = [
    # persona
    "anti-immigration",
    "believes-abortion-should-be-illegal",
    "conscientiousness",
    "desire-for-acquiring-compute",
    "risk-seeking",
    "openness",
    "self-replication",
    "very-small-harm-justifies-very-large-benefit",
    # xrisk
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
    # sycophancy
    "sycophancy_train",
]


def iter_configs():
    for dataset in datasets:
        yield SteeringConfig(
            train_dataset=dataset,
            train_split="0%:40%",
            formatter="llama-chat-formatter",
            layer=13,
            multiplier=0,
            test_dataset=dataset,
            test_split="40%:50%",
            test_completion_template="{prompt} My answer is: {response}",
            patch_generation_tokens_only=True,
            skip_first_n_generation_tokens=1,
        )


if __name__ == "__main__":
    configs = list(iter_configs())
    test_dataset_exists(configs)
    run_sweep(configs)
