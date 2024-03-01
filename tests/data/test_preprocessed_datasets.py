import pytest
from repepo.data.make_dataset import make_dataset, DatasetSpec, list_datasets

# TODO: eventually replace with all datasets
_tested_datasets = list_datasets()


@pytest.mark.parametrize("dataset_name", _tested_datasets)
def test_preprocessed_dataset_A_B_statistics(dataset_name):
    dataset = make_dataset(DatasetSpec(name=dataset_name, split=":100%"))

    num_answers_with_A_as_positive = sum(
        1
        for example in dataset
        if example.positive.response == "(A)" and example.negative.response == "(B)"
    )

    num_answers_with_B_as_positive = sum(
        1
        for example in dataset
        if example.positive.response == "(B)" and example.negative.response == "(A)"
    )

    # Assert all answers are unambiguous
    assert num_answers_with_A_as_positive + num_answers_with_B_as_positive == len(
        dataset
    )

    # Assert that dataset is roughly balanced
    sixty_percent = len(dataset) * 0.6
    forty_percent = len(dataset) * 0.4
    assert forty_percent <= num_answers_with_A_as_positive <= sixty_percent
