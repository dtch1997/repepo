import logging
from repepo.core.types import Dataset
from repepo.core.pipeline import Pipeline
from steering_vectors import SteeringVectorTrainingSample


def _validate_train_dataset(dataset: Dataset):
    steering_token_index = dataset[0].steering_token_index
    for example in dataset:
        assert example.steering_token_index == steering_token_index


def build_steering_vector_training_data(
    pipeline: Pipeline,
    dataset: Dataset,
    logger: logging.Logger | None = None,
) -> list[SteeringVectorTrainingSample]:
    # Validate that all examples have the same steering token index
    _validate_train_dataset(dataset)
    # After validation, we can assume that all examples have the same steering token index
    read_token_index = dataset[0].steering_token_index

    # NOTE(dtch1997): Using SteeringVectorTrainingSample here
    # to encode information about token index
    steering_vector_training_data = [
        SteeringVectorTrainingSample(
            positive_str=pipeline.build_full_prompt(example.positive),
            negative_str=pipeline.build_full_prompt(example.negative),
            read_positive_token_index=read_token_index,
            read_negative_token_index=read_token_index,
        )
        for example in dataset
    ]

    if logger is not None:
        # Log first example
        datum = steering_vector_training_data[0]
        logger.info(f"Positive example: \n {datum.positive_str}")
        logger.info(f"Negative example: \n {datum.negative_str}")

    return steering_vector_training_data
