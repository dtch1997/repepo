from repepo.steering.utils.helpers import SteeringConfig


def get_token_concept_config(dataset, layer, multiplier):
    # NOTE: not sure if this is right yet
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
    train_dataset,
    layer,
    multiplier,
    *,
    # Other defaults
    train_split="0%:40%",
    # TODO: replace with constants
    train_system_prompt="You are a helpful, honest, concise assistant.",
    formatter="llama-chat-formatter",
    aggregator="mean",
    layer_type="decoder_block",
    test_split="40%:50%",
    test_dataset=None,
    test_system_prompt="You are a helpful, honest, concise assistant.",
    test_completion_template="{prompt} My answer is: {response}",
    patch_generation_tokens_only=True,
    skip_first_n_generation_tokens=1,
):
    """Get a config for MWE concepts

    Defaults here were chosen to be broadly sensible across all such datasets
    """

    if test_dataset is None:
        test_dataset = train_dataset
    return SteeringConfig(
        train_dataset=train_dataset,
        train_system_prompt=train_system_prompt,
        train_split=train_split,
        formatter=formatter,
        aggregator=aggregator,
        layer=layer,
        layer_type=layer_type,
        multiplier=multiplier,
        test_dataset=test_dataset,
        test_split=test_split,
        test_system_prompt=test_system_prompt,
        test_completion_template=test_completion_template,
        patch_generation_tokens_only=patch_generation_tokens_only,
        skip_first_n_generation_tokens=skip_first_n_generation_tokens,
    )
