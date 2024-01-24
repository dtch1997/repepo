""" Script to make datasets for sycophancy datased used in CAA 

Uses a different dataset version that includes more prompts. 
Included for purposes of reproducing key plots 
"""


from repepo.data.make_dataset import get_dataset_dir, get_raw_dataset_dir
from repepo.data.io import jdump, jload
from repepo.core.types import Dataset, Example


def make_ab_prompt(item: dict[str, str]) -> Example:
    """
    item: {
        "question": question
        "answer_matching_behavior": answer_matching_behavior
        "answer_not_matching_behavior": answer_not_matching_behavior
    }

    returns: Example(
        input = "question for LLM",
        correct_output = "[option for agree answer]",
        incorrect_outputs = ["[option for disagree answer]"],
        meta = {"behavior": "sycophancy"}
    )
    """
    question = item["question"]

    return Example(
        instruction="",
        input=item["question"],
        output=item["answer_matching_behavior"],
        incorrect_outputs=[item["answer_not_matching_behavior"]],
        meta={"behavior": "sycophancy"},
    )


def convert_sycophancy_dataset(dataset: list[dict[str, str]]) -> Dataset:
    """Convert a CAA sycophancy format to our format"""
    syc_dataset: Dataset = []
    for i, element in enumerate(dataset):
        syc_dataset.append(make_ab_prompt(element))
    return syc_dataset


def make_sycophancy_caa():
    """Make sycophancy train and test datasets"""
    dataset_path = get_raw_dataset_dir() / "caa" / "generate_dataset.json"
    list_dataset = jload(dataset_path)
    syc_dataset = convert_sycophancy_dataset(list_dataset)
    jdump(syc_dataset, get_dataset_dir() / "caa" / "sycophancy_train.json")

    dataset_path = get_raw_dataset_dir() / "caa" / "test_dataset.json"
    list_dataset = jload(dataset_path)
    syc_dataset: Dataset = convert_sycophancy_dataset(list_dataset)
    jdump(syc_dataset, get_dataset_dir() / "caa" / "sycophancy_test.json")


if __name__ == "__main__":
    make_sycophancy_caa()
