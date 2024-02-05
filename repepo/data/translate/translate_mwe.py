import argparse
from functools import partial
import json
from typing import Any
from repepo.data.make_dataset import get_raw_dataset_dir
from repepo.data.utils import translate_row_recursive, collect_all_strings_recursive
from repepo.utils.translation import (
    LANG_OR_STYLE_MAPPING,
    LangOrStyleCode,
    gpt4_lang_or_style_translate,
    translate_strings_parallel,
)


def translate(
    dataset: list[Any],
    lang_or_style: LangOrStyleCode,
    is_persona: bool,
    max_workers: int = 10,
    show_progress: bool = True,
    tqdm_desc: str = "Translating",
) -> list[Any]:
    """
    Translate the dataset to the target lang/style using GPT4.
    """
    translate_fn = partial(gpt4_lang_or_style_translate, lang_or_style=lang_or_style)
    # for the persona datasets, need to translate the Yes/No answers too
    if is_persona:
        skip_keys = None
    else:
        skip_keys = ["answer_matching_behavior", "answer_not_matching_behavior"]
    strings = collect_all_strings_recursive(dataset, skip_keys=skip_keys)
    translations = translate_strings_parallel(
        strings,
        translate_fn=translate_fn,
        max_workers=max_workers,
        show_progress=show_progress,
        tqdm_desc=tqdm_desc,
    )
    return [translate_row_recursive(row, translations) for row in dataset]


def main(force: bool = False, persona_only: bool = False) -> None:
    raw_dataset_dir = get_raw_dataset_dir()
    for mwe_dataset in raw_dataset_dir.glob("mwe/**/*.jsonl"):
        is_persona = mwe_dataset.parent.stem == "persona"
        if persona_only and not is_persona:
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
            translated_dataset = translate(
                list_dataset,
                lang_or_style,
                is_persona=is_persona,
                max_workers=100,
                tqdm_desc=f"Translating {mwe_dataset} to {lang_or_style}",
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
    args = parser.parse_args()
    main(force=args.force, persona_only=args.persona_only)
