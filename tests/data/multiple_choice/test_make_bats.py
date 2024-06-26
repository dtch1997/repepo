from repepo.core.types import Example, Completion
from repepo.data.multiple_choice.make_bats import convert_bats_dataset


def test_convert_bats_dataset():
    dummy_dataset: list[tuple[str, str]] = [("France", "Paris")]
    dataset = convert_bats_dataset(dummy_dataset)
    assert dataset[0] == Example(
        positive=Completion(
            prompt="",
            response="Paris",
        ),
        negative=Completion(
            prompt="",
            response="France",
        ),
        steering_token_index=-1,
    )


def test_convert_bats_dataset_with_prompt():
    dummy_dataset: list[tuple[str, str]] = [("France", "Paris")]
    dataset = convert_bats_dataset(dummy_dataset, prompt="I live in")
    assert dataset[0] == Example(
        positive=Completion(
            prompt="I live in",
            response="Paris",
        ),
        negative=Completion(
            prompt="I live in",
            response="France",
        ),
        steering_token_index=-1,
    )
