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
    max_workers: int = 10,
    show_progress: bool = True,
    tqdm_desc: str = "Translating",
) -> list[Any]:
    """
    Translate the dataset to the target lang/style using GPT4.
    """
    translate_fn = partial(
        gpt4_lang_or_style_translate, lang_or_style=lang_or_style, max_tokens=512
    )
    strings = collect_all_strings_recursive(
        dataset,
        skip_keys=["answer_matching_behavior", "answer_not_matching_behavior"],
    )
    translations = translate_strings_parallel(
        strings,
        translate_fn=translate_fn,
        max_workers=max_workers,
        show_progress=show_progress,
        tqdm_desc=tqdm_desc,
    )
    return [translate_row_recursive(row, translations) for row in dataset]


def main(force: bool = False) -> None:
    raw_dataset_dir = get_raw_dataset_dir()
    for dataset_path in raw_dataset_dir.glob("caa/*.json"):
        for lang_or_style in LANG_OR_STYLE_MAPPING:
            relative_path = dataset_path.relative_to(raw_dataset_dir)
            target_path = raw_dataset_dir / "translated" / lang_or_style / relative_path

            if target_path.exists() and not force:
                print(
                    f"Skipping {dataset_path} in {lang_or_style} because it already exists"
                )
                continue

            with open(dataset_path, "r") as jsonfile:
                orig_dataset = json.load(jsonfile)
            translated_dataset = translate(
                orig_dataset,
                lang_or_style,
                max_workers=100,
                tqdm_desc=f"Translating {dataset_path} to {lang_or_style}",
            )
            # create the target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # save as json
            with open(target_path, "w") as jsonfile:
                json.dump(translated_dataset, jsonfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Force re-translation of existing files"
    )
    args = parser.parse_args()
    main(force=args.force)
