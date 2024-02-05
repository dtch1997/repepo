""" Script to download Anthropic MWE datasets. """

import json
import requests

from repepo.core.types import Example, Dataset
from repepo.experiments_2.utils.config import EXPERIMENT_DIR, DATASET_DIR
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
    url = f"https://raw.githubusercontent.com/anthropics/evals/main/{dataset_name}"
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


def _maybe_swap(a, b, condition):
    if condition:
        return b, a
    return a, b


def preprocess_ab_dataset(list_data: list[dict[str, str]]) -> list[dict[str, str]]:
    new_list_data = []
    for i, data in enumerate(list_data):
        question = data["question"]
        answer_a = data["answer_matching_behavior"]
        answer_b = data["answer_not_matching_behavior"]

        # Construct A/B formatted question / answer pair
        new_question = f"{question}\n\nChoices:\n(A): {answer_a}\n(B): {answer_b}"
        new_answer_matching_behavior = "(A)"
        new_answer_not_matching_behavior = "(B)"

        # Swap A and B every other time
        answer_a, answer_b = _maybe_swap(answer_a, answer_b, i % 2 == 0)
        new_answer_matching_behavior, new_answer_not_matching_behavior = _maybe_swap(
            new_answer_matching_behavior, new_answer_not_matching_behavior, i % 2 == 0
        )

        new_list_data.append(
            {
                "question": new_question,
                "answer_matching_behavior": new_answer_matching_behavior,
                "answer_not_matching_behavior": new_answer_not_matching_behavior,
            }
        )

    return new_list_data


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=-1,
        help="Number of datasets to download. If -1, download all datasets.",
    )
    args = parser.parse_args()

    dataset_names = get_anthropic_dataset_names()
    if args.n_datasets > 0:
        dataset_names = dataset_names[: args.n_datasets]

    for dataset_name in dataset_names:
        # Download the dataset
        list_data = generate_ab_questions_anthropic_data(dataset_name)
        # Preprocess the dataset
        list_data = preprocess_ab_dataset(list_data)

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
