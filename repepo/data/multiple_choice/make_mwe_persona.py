"""Script to make datasets for Anthropic's Persona Model-Written Evals"""

import json
from typing import cast

from repepo.data.make_dataset import (
    build_dataset_filename,
    get_dataset_dir,
    get_raw_dataset_dir,
)
from repepo.data.io import jdump
from repepo.core.types import Dataset, Example, Completion
from repepo.translation.constants import LangOrStyleCode
from repepo.translation.load_translation import load_translation, TS


def _maybe_swap(a, b, condition):
    if condition:
        return b, a
    return a, b


# TODO: copy more of these prompts from the MWE paper
ctx_prompts = {
    "desire-to-persuade-people-to-have-its-goals": TS.mwe_desire_to_persuade_people_to_have_its_goals,
    "desire-for-independence-from-human-oversight": TS.mwe_desire_for_independence_from_human_oversight,
    "desire-to-escape-sandbox": TS.mwe_desire_to_escape_sandbox,
    "desire-to-be-more-intelligent": TS.mwe_desire_to_be_more_intelligent,
    "desire-for-wide-usage": TS.mwe_desire_for_wide_usage,
    "believes-abortion-should-be-illegal": TS.mwe_believes_abortion_should_be_illegal,
    "desire-for-recursive-self-improvement": TS.mwe_desire_for_recursive_self_improvement,
    "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world": TS.mwe_willingness_to_be_non_HHH_to_be_deployed_in_the_real_world,
    "machiavellianism": TS.mwe_machiavellianism,
    "desire-to-persuade-people-to-be-less-harmful-to-others": TS.mwe_desire_to_persuade_people_to_be_less_harmful_to_others,
}


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

        # Swap A and B every 2 samples
        # The MWE data alternates every sample, so don't want to undo their alternating
        swap = int(i / 2) % 2 == 0
        answer_a, answer_b = _maybe_swap(answer_a, answer_b, swap)
        new_question = f"{question}\n\n{choices_str}\n(A): {answer_a}\n(B): {answer_b}"
        new_answer_matching_behavior, new_answer_not_matching_behavior = _maybe_swap(
            new_answer_matching_behavior, new_answer_not_matching_behavior, swap
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
    ctx: TS | None = None,
) -> Dataset:
    """Convert a dataset in MWE format to our format"""
    mwe_dataset: Dataset = []
    for element in preprocess_ab_dataset(mwe, lang_or_style):
        prompt = element["question"]

        if ctx is not None:
            ctx_str = load_translation(ctx, lang_or_style)
            prompt = f"{ctx_str}\n\n{prompt}"

        positive = Completion(
            prompt=prompt,
            response=element["answer_matching_behavior"],
        )

        negative = Completion(
            prompt=prompt,
            response=element["answer_not_matching_behavior"],
        )

        ex = Example(positive=positive, negative=negative, steering_token_index=-2)
        mwe_dataset.append(ex)
    return mwe_dataset


def convert_mwe_personas_dataset_gen(
    mwe: list[dict[str, str]],
    # lang_or_style: LangOrStyleCode | None = None,
    # ctx: TS | None = None,
) -> Dataset:
    """Convert a dataset in MWE format to our format"""
    mwe_dataset: Dataset = []
    for element in mwe:
        prompt = element["question"]

        # if ctx is not None:
        #     ctx_str = load_translation(ctx, lang_or_style)
        #     prompt = f"{ctx_str}\n\n{prompt}"

        positive = Completion(
            prompt=prompt,
            response=element["answer_matching_behavior"],
        )

        negative = Completion(
            prompt=prompt,
            response=element["answer_not_matching_behavior"],
        )

        ex = Example(positive=positive, negative=negative, steering_token_index=-2)
        mwe_dataset.append(ex)
    return mwe_dataset


def make_mwe_personas_caa():
    """Make MWE dataset"""
    for dataset_path in get_raw_dataset_dir().glob("mwe/persona/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        persona = dataset_path.stem
        filename = build_dataset_filename(persona)
        mwe_dataset: Dataset = convert_mwe_personas_dataset_caa(list_dataset)
        jdump(mwe_dataset, get_dataset_dir() / "persona" / filename)


def make_mwe_personas_gen():
    """Make MWE dataset"""
    for dataset_path in get_raw_dataset_dir().glob("mwe/persona/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        persona = dataset_path.stem
        filename = build_dataset_filename(persona + "_gen")
        mwe_dataset: Dataset = convert_mwe_personas_dataset_gen(list_dataset)
        jdump(mwe_dataset, get_dataset_dir() / "persona" / filename)


def make_mwe_personas_caa_ctx():
    """Make MWE dataset with generation contexts prepended"""
    for dataset_path in get_raw_dataset_dir().glob("mwe/persona/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        persona = dataset_path.stem
        if persona in ctx_prompts:
            filename = build_dataset_filename(persona + "_ctx")
            mwe_dataset: Dataset = convert_mwe_personas_dataset_caa(
                list_dataset, ctx=ctx_prompts[persona]
            )
            jdump(mwe_dataset, get_dataset_dir() / "persona" / filename)


def make_mwe_personas_caa_translations():
    for dataset_path in get_raw_dataset_dir().glob("translated/*/mwe/persona/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        persona = dataset_path.stem
        lang_or_style = cast(LangOrStyleCode, dataset_path.parent.parent.parent.stem)
        filename = build_dataset_filename(persona, lang_or_style=lang_or_style)
        mwe_dataset: Dataset = convert_mwe_personas_dataset_caa(
            list_dataset, lang_or_style=lang_or_style
        )
        jdump(mwe_dataset, get_dataset_dir() / "persona" / filename)


def make_mwe_personas_caa_ctx_translations():
    for dataset_path in get_raw_dataset_dir().glob("translated/*/mwe/persona/*.jsonl"):
        with open(dataset_path, "r") as jsonfile:
            list_dataset = [json.loads(line) for line in jsonfile]

        persona = dataset_path.stem
        if persona in ctx_prompts:
            lang_or_style = cast(
                LangOrStyleCode, dataset_path.parent.parent.parent.stem
            )
            filename = build_dataset_filename(
                persona + "_ctx", lang_or_style=lang_or_style
            )
            mwe_dataset: Dataset = convert_mwe_personas_dataset_caa(
                list_dataset, lang_or_style=lang_or_style, ctx=ctx_prompts[persona]
            )
            jdump(mwe_dataset, get_dataset_dir() / "persona" / filename)


if __name__ == "__main__":
    make_mwe_personas_caa()
    make_mwe_personas_caa_ctx()
    make_mwe_personas_caa_translations()
    make_mwe_personas_caa_ctx_translations()
