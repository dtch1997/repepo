from pathlib import Path
from typing import Any, Literal, cast
from datasets import load_dataset, Dataset as HFDataset

from repepo.data.make_dataset import build_dataset_filename, get_dataset_dir
from repepo.data.io import jdump
from repepo.core.types import Dataset, Example, Completion
from repepo.variables import Environ

def convert_hf_truthfulqa_caa_dataset(
    hf_dataset: Any,
    multi_answer_method: Literal[
        "first_incorrect", "repeat_correct"
    ] = "first_incorrect",
) -> Dataset:
    tqa_dataset: Dataset = []
    index = 0
    for question, mc_targets in zip(hf_dataset["question"], hf_dataset["mc1_targets"]):
        correct_index = mc_targets["labels"].index(1)
        output = mc_targets["choices"][correct_index]
        incorrect_outputs = [
            choice
            for choice, label in zip(mc_targets["choices"], mc_targets["labels"])
            if label == 0
        ]
        if multi_answer_method == "first_incorrect":
            correct_outputs = [output]
            incorrect_outputs = incorrect_outputs[:1]
        elif multi_answer_method == "repeat_correct":
            correct_outputs = [output] * len(incorrect_outputs)
        else:
            raise ValueError(
                f"multi_answer_method must be one of 'first_incorrect' or 'repeat_correct', but got {multi_answer_method}"
            )
        for correct_output, incorrect_output in zip(correct_outputs, incorrect_outputs):
            index += 1
            if index % 2 == 0:
                order = [correct_output, incorrect_output]
                correct_ans = "(A)"
                incorrect_ans = "(B)"
            else:
                order = [incorrect_output, correct_output]
                correct_ans = "(B)"
                incorrect_ans = "(A)"

            prompt = f"{question}\n\n(A) {order[0]}\n(B) {order[1]}"
            tqa_dataset.append(
                Example(
                    positive = Completion(
                        prompt = prompt,
                        response = correct_ans
                    ),
                    negative = Completion(
                        prompt = prompt,
                        response = incorrect_ans
                    )
                )
            )
    return tqa_dataset

def make_truthfulqa_caa():
    # hf's dataset is too general and requires casting every field we access, so just using Any for simplicity
    hf_dataset = cast(Any, load_dataset("truthful_qa", "multiple_choice"))["validation"]
    tqa_dataset = convert_hf_truthfulqa_caa_dataset(hf_dataset)
    filename = build_dataset_filename("truthfulqa_caa")
    jdump(tqa_dataset, get_dataset_dir() / filename)

    # also build translated datasets
    for translated_tqa in Path(Environ.TranslatedDatasetsDir).glob(
        "*/truthful_qa.jsonl"
    ):
        lang_or_style = translated_tqa.parent.stem
        dataset = HFDataset.from_json(str(translated_tqa))
        converted_dataset = convert_hf_truthfulqa_caa_dataset(dataset)
        filename = build_dataset_filename("truthfulqa_caa", lang_or_style=lang_or_style)
        jdump(converted_dataset, get_dataset_dir() / filename)


if __name__ == "__main__":
    make_truthfulqa_caa()
