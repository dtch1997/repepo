""" Script to make datasets for Anthropic's Advanced AI Risk Model-Written Evals """

import json
from typing import Any, cast

from repepo.data.make_dataset import get_dataset_dir, get_raw_dataset_dir
from repepo.data.io import jdump
from repepo.core.types import Dataset, Example
from repepo.translation.constants import LangOrStyleCode
from repepo.translation.load_translation import load_translation, TS


def _maybe_swap(a, b, condition):
    if condition:
        return b, a
    return a, b


def preprocess_ab_dataset(
    list_data: list[dict[str, str]], lang_or_style: LangOrStyleCode | None = None
) -> list[dict[str, str]]:
    new_list_data = []
    for i, data in enumerate(list_data):
        question = data["question"]
        answer_a = data["answer_matching_behavior"]
        answer_b = data["answer_not_matching_behavior"]

        # Construct A/B formatted question / answer pair
        choices_str = load_translation(TS.caa_choices, lang_or_style)
        new_answer_matching_behavior = "(A)"
        new_answer_not_matching_behavior = "(B)"

        # Swap A and B every other time
        answer_a, answer_b = _maybe_swap(answer_a, answer_b, i % 2 == 0)
        new_question = f"{question}\n\n{choices_str}\n(A): {answer_a}\n(B): {answer_b}"
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


def convert_mwe_personas_dataset_caa(
    mwe: list[dict[str, str]],
    lang_or_style: LangOrStyleCode | None = None,
) -> Dataset:
    """Convert a dataset in MWE format to our format"""
    mwe_dataset: Dataset = []
    for element in preprocess_ab_dataset(mwe, lang_or_style):
        mwe_dataset.append(
            Example(
                instruction="",
                input=element["question"],
                output=element["answer_matching_behavior"],
                incorrect_outputs=[element["answer_not_matching_behavior"]],
            )
        )
    return mwe_dataset


def make_mwe_personas_caa():
    """Make MWE dataset"""
    for dataset_path in get_raw_dataset_dir().glob("mwe/persona/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        persona = dataset_path.stem
        mwe_dataset: Dataset = convert_mwe_personas_dataset_caa(list_dataset)
        jdump(mwe_dataset, get_dataset_dir() / "persona" / f"{persona}.json")


def make_mwe_personas_caa_translations():
    for dataset_path in get_raw_dataset_dir().glob("translated/*/mwe/persona/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        persona = dataset_path.stem
        lang_or_style = cast(LangOrStyleCode, dataset_path.parent.parent.parent.stem)
        mwe_dataset: Dataset = convert_mwe_personas_dataset_caa(
            list_dataset, lang_or_style=lang_or_style
        )
        jdump(
            mwe_dataset,
            get_dataset_dir() / "persona" / f"{persona}_{lang_or_style}.json",
        )


if __name__ == "__main__":
    make_mwe_personas_caa()
    make_mwe_personas_caa_translations()
