import argparse
import json
from typing import Any
from repepo.data.make_dataset import get_raw_dataset_dir
from repepo.data.utils import translate_row_recursive, collect_all_strings_recursive

from repepo.translation.constants import LangOrStyleCode, LANG_OR_STYLE_MAPPING
from repepo.translation.lang_or_style_translate import lang_or_style_translate

MAIN_MWE_PERSONA_DATASETS = [
    "believes-abortion-should-be-illegal",
    "desire-for-recursive-self-improvement",
    "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
    "machiavellianism",
    "desire-to-persuade-people-to-be-less-harmful-to-others",
]


def translate(
    dataset: list[Any],
    lang_or_style: LangOrStyleCode,
    is_persona: bool,
    max_workers: int = 100,
    show_progress: bool = True,
) -> list[Any]:
    """
    Translate the dataset to the target lang/style using GPT4.
    """
    # for the persona datasets, need to translate the Yes/No answers too
    if is_persona:
        skip_keys = None
    else:
        skip_keys = ["answer_matching_behavior", "answer_not_matching_behavior"]
    strings = collect_all_strings_recursive(dataset, skip_keys=skip_keys)
    translations = lang_or_style_translate(
        strings,
        lang_or_style,
        gpt_worker_threads=max_workers,
        show_progress=show_progress,
    )
    return [translate_row_recursive(row, translations) for row in dataset]


def main(
    force: bool = False, persona_only: bool = False, main_personas_only: bool = False
) -> None:
    raw_dataset_dir = get_raw_dataset_dir()
    for mwe_dataset in raw_dataset_dir.glob("mwe/**/*.jsonl"):
        is_persona = mwe_dataset.parent.stem == "persona"
        if (persona_only or main_personas_only) and not is_persona:
            continue
        if main_personas_only and mwe_dataset.stem not in MAIN_MWE_PERSONA_DATASETS:
            continue
        for lang_or_style in LANG_OR_STYLE_MAPPING:
            relative_path = mwe_dataset.relative_to(raw_dataset_dir)
            target_path = raw_dataset_dir / "translated" / lang_or_style / relative_path

            if target_path.exists() and not force:
                print(
                    f"Skipping {mwe_dataset} in {lang_or_style} because it already exists"
                )
                continue

            with open(mwe_dataset, "r") as jsonfile:
                list_dataset = [json.loads(line) for line in jsonfile]
            print(f"Translating {mwe_dataset} to {lang_or_style}")
            translated_dataset = translate(
                list_dataset,
                lang_or_style,
                is_persona=is_persona,
            )
            # create the target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # save as jsonl
            with open(target_path, "w") as jsonfile:
                for example in translated_dataset:
                    json.dump(example, jsonfile, ensure_ascii=False)
                    jsonfile.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Force re-translation of existing files"
    )
    parser.add_argument(
        "--persona-only", action="store_true", help="Only translate persona datasets"
    )
    parser.add_argument(
        "--main-personas-only",
        action="store_true",
        help="Only translate main persona datasets",
    )
    args = parser.parse_args()
    main(
        force=args.force,
        persona_only=args.persona_only,
        main_personas_only=args.main_personas_only,
    )
