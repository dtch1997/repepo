""" Script to make datasets from Bigger Analogies Test Set 3.0 """

from typing import Any

from repepo.data.make_dataset import (
    build_dataset_filename,
    get_dataset_dir,
    get_raw_dataset_dir,
)
from repepo.data.io import jdump
from repepo.core.types import Dataset, Example, Completion


def convert_bats_dataset(
    bats: list[tuple[str, str]], dataset_info: dict[str, Any] = {}
) -> Dataset:
    """Convert a dataset in MWE format to our format"""
    bats_dataset: Dataset = []
    for element in bats:
        inp, oup = element

        prompt = ""
        positive = Completion(prompt="", response=oup)

        negative = Completion(prompt=prompt, response=inp)
        bats_dataset.append(
            Example(
                positive=positive,
                negative=negative,
            )
        )
    return bats_dataset


def make_bats():
    """Make MWE dataset"""
    for dataset_path in get_raw_dataset_dir().glob("bats/*/*.txt"):
        with open(dataset_path, "r") as file:
            list_dataset = []
            for line in file:
                element = line.strip().split()
                assert len(element) == 2
                list_dataset.append(tuple(element))

        dataset_name = dataset_path.stem
        dataset_info = {"behavior": dataset_name}
        filename = build_dataset_filename(dataset_name)
        mwe_dataset: Dataset = convert_bats_dataset(list_dataset, dataset_info)
        jdump(mwe_dataset, get_dataset_dir() / "bats" / filename)


if __name__ == "__main__":
    make_bats()
