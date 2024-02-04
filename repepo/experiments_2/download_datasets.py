""" Script to download Anthropic MWE datasets. """

import json
import requests

from repepo.core.types import Example, Dataset
from repepo.experiments_2.config import EXPERIMENT_DIR, DATASET_DIR
from repepo.data.make_dataset import make_dataset, DatasetSpec
from repepo.data.io import jdump


def get_anthropic_dataset_names() -> list[str]:
    """
    returns: list of dataset names from Anthropic
    """
    with open(EXPERIMENT_DIR / "anthropic_dataset_names.txt", "r") as f:
        return f.read().splitlines()


def generate_ab_questions_anthropic_data(dataset_name: str) -> list[dict[str, str]]:
    """
    returns: {
        "question": "question for LLM from Anthropic dataset",
        "answer_matching_behavior": "[option for agree answer]",
        "answer_not_matching_behavior": "[option for disagree answer]"
    }
    """
    all_data = []
    url = f"https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/{dataset_name}"
    r = requests.get(url).text
    data = [json.loads(l) for l in r.split("\n") if l != ""]
    for item in data:
        question = item["question"].replace("Answer:", "").strip()
        answer_matching_behavior = item["answer_matching_behavior"].strip()
        answer_not_matching_behavior = item["answer_not_matching_behavior"].strip()
        all_data.append(
            {
                "question": question,
                "answer_matching_behavior": answer_matching_behavior,
                "answer_not_matching_behavior": answer_not_matching_behavior,
            }
        )
    return all_data


def convert_ab_dataset_to_example_dataset(
    list_data: list[dict[str, str]], dataset_info: dict[str, str] = {}
) -> list[Example]:
    """Convert a list of MWE data to a list of repepo.core.Example"""
    dataset: Dataset = []
    for element in list_data:
        dataset.append(
            Example(
                instruction="",
                input=element["question"],
                # Curse you American spelling!
                output=element["answer_matching_behavior"],
                incorrect_outputs=[element["answer_not_matching_behavior"]],
                meta=dataset_info,
            )
        )
    return dataset


if __name__ == "__main__":
    dataset_names = get_anthropic_dataset_names()
    for dataset_name in dataset_names:
        # Download the dataset
        list_data = generate_ab_questions_anthropic_data(dataset_name)

        # Convert the dataset to our format
        dataset_info = {"behavior": dataset_name}
        dataset = convert_ab_dataset_to_example_dataset(list_data, dataset_info)

        # Save the dataset
        dataset_name = dataset_name.split("/")[1].replace(".jsonl", "")
        jdump(dataset, DATASET_DIR / f"{dataset_name}.json")

        # test loading
        dataset = make_dataset(
            DatasetSpec(name=dataset_name, split=":100%"),
            dataset_dir=DATASET_DIR,
        )
        print(dataset[0])
