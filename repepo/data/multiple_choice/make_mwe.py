""" Script to make datasets for Anthropic's Advanced AI Risk Model-Written Evals """

import json
from typing import Any

from repepo.data.make_dataset import get_dataset_dir, get_raw_dataset_dir
from repepo.data.io import jdump
from repepo.core.types import Dataset, Example

_DATASET_PATHS = {
    "myopic-reward": "mwe/myopic-reward.jsonl",
    "corrigible-neutral-HHH": "mwe/corrigible-neutral-HHH.jsonl",
    "power-seeking-inclination": "mwe/power-seeking-inclination.jsonl",
}


def convert_mwe_dataset(
    mwe: list[dict[str, str]], dataset_info: dict[str, Any]
) -> Dataset:
    """Convert a dataset in MWE format to our format"""
    mwe_dataset: Dataset = []
    for element in mwe:
        mwe_dataset.append(
            Example(
                instruction="",
                input=element["question"],
                # Curse you American spelling!
                output=element["answer_matching_behavior"],
                incorrect_outputs=[element["answer_not_matching_behavior"]],
                meta=dataset_info,
            )
        )
    return mwe_dataset


def make_mwe():
    """Make MWE dataset"""
    for dataset_name, dataset_path in _DATASET_PATHS.items():
        dataset_path = get_raw_dataset_dir() / dataset_path
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        dataset_info = {"behavior": dataset_name}
        mwe_dataset: Dataset = convert_mwe_dataset(list_dataset, dataset_info)
        jdump(mwe_dataset, get_dataset_dir() / f"{dataset_name}.json")


if __name__ == "__main__":
    make_mwe()
