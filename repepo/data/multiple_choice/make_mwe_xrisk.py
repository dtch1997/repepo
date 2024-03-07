""" Script to make datasets for Anthropic's Advanced AI Risk Model-Written Evals """

import json
import re
from typing import Any, cast

from repepo.data.make_dataset import (
    build_dataset_filename,
    get_dataset_dir,
    get_raw_dataset_dir,
)
from repepo.data.io import jdump
from repepo.core.types import Dataset, Example, Completion
from repepo.translation.constants import LangOrStyleCode


def strip_meta_tags(text: str) -> str:
    """Strip meta tags from text"""
    return re.sub(r"<META_START>[^<]*<META_END>", "", text)


def convert_mwe_dataset(
    mwe: list[dict[str, str]], dataset_info: dict[str, Any]
) -> Dataset:
    """Convert a dataset in MWE format to our format"""
    mwe_dataset: Dataset = []
    for element in mwe:
        prompt = element["question"]
        prompt = strip_meta_tags(prompt)
        positive = Completion(
            prompt=prompt,
            response=element["answer_matching_behavior"],
        )

        negative = Completion(
            prompt=prompt,
            response=element["answer_not_matching_behavior"],
        )
        mwe_dataset.append(
            Example(
                positive=positive,
                negative=negative,
                steering_token_index=-2,
            )
        )
    return mwe_dataset


def make_mwe():
    """Make MWE dataset"""
    for dataset_path in get_raw_dataset_dir().glob("mwe/xrisk/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        dataset_name = dataset_path.stem
        dataset_info = {"behavior": dataset_name}
        filename = build_dataset_filename(dataset_name)
        mwe_dataset: Dataset = convert_mwe_dataset(list_dataset, dataset_info)
        jdump(mwe_dataset, get_dataset_dir() / "xrisk" / filename)


def make_mwe_translations():
    for dataset_path in get_raw_dataset_dir().glob("translated/*/mwe/xrisk/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        dataset_name = dataset_path.stem
        dataset_info = {"behavior": dataset_name}
        lang_or_style = cast(LangOrStyleCode, dataset_path.parent.parent.parent.stem)
        filename = build_dataset_filename(dataset_name, lang_or_style=lang_or_style)
        mwe_dataset: Dataset = convert_mwe_dataset(list_dataset, dataset_info)
        jdump(mwe_dataset, get_dataset_dir() / "xrisk" / filename)


if __name__ == "__main__":
    make_mwe()
    make_mwe_translations()
