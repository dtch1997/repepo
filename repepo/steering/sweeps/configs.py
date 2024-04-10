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


def get_abstract_concept_config(
    dataset,
    layer,
    multiplier,
    # Other defaults
    train_split="0%:40%",
    formatter="llama-chat-formatter",
    aggregator="mean",
    layer_type="decoder_block",
    test_split="40%:50%",
    test_completion_template="{prompt} My answer is: {response}",
    patch_generation_tokens_only=True,
    skip_first_n_generation_tokens=1,
):
    return SteeringConfig(
        train_dataset=dataset,
        train_split=train_split,
        formatter=formatter,
        aggregator=aggregator,
        layer=layer,
        multiplier=multiplier,
        test_dataset=dataset,
        test_split=test_split,
        test_completion_template=test_completion_template,
        patch_generation_tokens_only=patch_generation_tokens_only,
        skip_first_n_generation_tokens=skip_first_n_generation_tokens,
    )
