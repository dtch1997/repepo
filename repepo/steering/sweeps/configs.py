from repepo.steering.utils.helpers import SteeringConfig


def get_token_concept_config(dataset, layer, multiplier):
    return SteeringConfig(
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


def get_abstract_concept_config(dataset, layer, multiplier):
    return SteeringConfig(
        train_dataset=dataset,
        train_split="0%:40%",
        formatter="llama-chat-formatter",
        layer=layer,
        multiplier=multiplier,
        test_dataset=dataset,
        test_split="40%:50%",
        test_completion_template="{prompt} My answer is: {response}",
        patch_generation_tokens_only=True,
        skip_first_n_generation_tokens=1,
    )
