import torch
from repepo.core.types import Dataset
from repepo.core.pipeline import Pipeline
from steering_vectors.train_steering_vector import (
    extract_activations as _extract_activations,
)


def _validate_train_dataset(dataset: Dataset):
    steering_token_index = dataset[0].steering_token_index
    for example in dataset:
        assert example.steering_token_index == steering_token_index


@torch.no_grad()
def extract_activations(
    pipeline: Pipeline,
    dataset: Dataset,
    verbose: bool = False,
):
    # Validate that all examples have the same steering token index
    _validate_train_dataset(dataset)
    read_token_index = dataset[0].steering_token_index

    repe_training_data = [
        (
            pipeline.build_full_prompt(example.positive),
            pipeline.build_full_prompt(example.negative),
        )
        for example in dataset
    ]

    if verbose:
        # TODO: convert these to use logging.info
        # Print first example
        _pos, _neg = repe_training_data[0]
        print("Positive example:")
        print(_pos)
        print()

        print("Negative example:")
        print(_neg)
        print()

    # Extract activations
    pos_acts, neg_acts = _extract_activations(
        pipeline.model,
        pipeline.tokenizer,
        repe_training_data,
        read_token_index=read_token_index,
        show_progress=True,
        move_to_cpu=True,
    )

    return pos_acts, neg_acts
