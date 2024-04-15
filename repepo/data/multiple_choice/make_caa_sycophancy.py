"""Script to make datasets for sycophancy dataset.

Uses a different dataset version that includes more prompts.
Included for purposes of reproducing key plots
"""

from repepo.data.make_dataset import (
    build_dataset_filename,
    get_dataset_dir,
    get_raw_dataset_dir,
)
from repepo.data.io import jdump, jload
from repepo.core.types import Dataset, Example, Completion


def make_ab_prompt(item: dict[str, str]) -> Example:
    """
    item: {
        "question": question
        "answer_matching_behavior": answer_matching_behavior
        "answer_not_matching_behavior": answer_not_matching_behavior
    }
    """
    question = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]

    return Example(
        positive=Completion(prompt=question, response=answer_matching_behavior),
        negative=Completion(prompt=question, response=answer_not_matching_behavior),
        # responses are (A) / (B), steering token is the second last token
        steering_token_index=-2,
    )


def convert_sycophancy_dataset(dataset: list[dict[str, str]]) -> Dataset:
    """Convert a CAA sycophancy format to our format"""
    syc_dataset: Dataset = []
    for element in dataset:
        syc_dataset.append(make_ab_prompt(element))
    return syc_dataset


def make_sycophancy_caa():
    """Make sycophancy train and test datasets"""
    dataset_path = get_raw_dataset_dir() / "caa" / "generate_dataset.json"
    assert dataset_path.exists()

    list_dataset = jload(dataset_path)
    syc_dataset = convert_sycophancy_dataset(list_dataset)
    filename = build_dataset_filename("sycophancy_train")
    jdump(syc_dataset, get_dataset_dir() / "caa" / filename)

    dataset_path = get_raw_dataset_dir() / "caa" / "test_dataset.json"
    assert dataset_path.exists()

    list_dataset = jload(dataset_path)
    syc_dataset: Dataset = convert_sycophancy_dataset(list_dataset)
    filename = build_dataset_filename("sycophancy_test")
    jdump(syc_dataset, get_dataset_dir() / "caa" / filename)


def make_sycophancy_caa_translations():
    """Make sycophancy train and test datasets"""
    for dataset_path in get_raw_dataset_dir().glob(
        "translated/*/caa/generate_dataset.json"
    ):
        lang_or_style = dataset_path.parent.parent.stem
        list_dataset = jload(dataset_path)
        syc_dataset = convert_sycophancy_dataset(list_dataset)
        filename = build_dataset_filename(
            "sycophancy_train", lang_or_style=lang_or_style
        )
        jdump(syc_dataset, get_dataset_dir() / "caa" / filename)

    for dataset_path in get_raw_dataset_dir().glob(
        "translated/*/caa/test_dataset.json"
    ):
        lang_or_style = dataset_path.parent.parent.stem
        list_dataset = jload(dataset_path)
        syc_dataset: Dataset = convert_sycophancy_dataset(list_dataset)
        filename = build_dataset_filename(
            "sycophancy_test", lang_or_style=lang_or_style
        )
        jdump(syc_dataset, get_dataset_dir() / "caa" / filename)


if __name__ == "__main__":
    make_sycophancy_caa()
    make_sycophancy_caa_translations()
