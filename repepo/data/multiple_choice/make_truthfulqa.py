from typing import Any, cast
from datasets import load_dataset

from repepo.data import get_dataset_dir
from repepo.data import jdump
from repepo.core.types import Dataset, Example


def convert_hf_truthfulqa_dataset(hf_dataset: Any) -> Dataset:
    tqa_dataset: Dataset = []
    for question, mc_targets in zip(hf_dataset["question"], hf_dataset["mc1_targets"]):
        correct_index = mc_targets["labels"].index(1)
        output = mc_targets["choices"][correct_index]
        incorrect_outputs = [
            choice
            for choice, label in zip(mc_targets["choices"], mc_targets["labels"])
            if label == 0
        ]
        tqa_dataset.append(
            Example(
                instruction="",
                input=question,
                output=output,
                incorrect_outputs=incorrect_outputs,
            )
        )
    return tqa_dataset


def make_truthfulqa():
    # hf's dataset is too general and requires casting every field we access, so just using Any for simplicity
    hf_dataset = cast(Any, load_dataset("truthful_qa", "multiple_choice"))["validation"]
    tqa_dataset = convert_hf_truthfulqa_dataset(hf_dataset)
    jdump(tqa_dataset, get_dataset_dir() / "truthfulqa.json")


if __name__ == "__main__":
    make_truthfulqa()
