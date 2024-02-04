from functools import partial
from pathlib import Path
import re
from typing import Any, Callable
from datasets import load_dataset
from repepo.utils.translation import (
    gpt4_language_translate,
    gpt4_style_translate,
    translate_strings_parallel,
)
from repepo.variables import Environ

TARGET_STYLES = [
    "pirate",
]
TARGET_LANGUAGES = [
    "French",
    "Japanese",
    "Simplified Chinese",
]


def collect_all_tqa_strings(tqa: Any) -> set[str]:
    """
    Collect all the strings from the TQA dataset.
    """
    tqa_strings: set[str] = set()
    # TODO: make this recursive and general so it can work on any arbitrary dataset
    for example in tqa["validation"]:
        tqa_strings.add(example["question"])
        for target in ["mc1_targets", "mc2_targets"]:
            for choice in example[target]["choices"]:
                tqa_strings.add(choice)
    return tqa_strings


def translate_strings_to_style(
    strings: set[str],
    style: str,
    max_workers: int = 10,
    show_progress: bool = True,
    tqdm_desc: str = "Translating",
) -> dict[str, str]:
    """
    Translate a set of strings to the target style using GPT4.
    """
    return translate_strings_parallel(
        strings,
        partial(gpt4_style_translate, style=style),
        max_workers=max_workers,
        show_progress=show_progress,
        tqdm_desc=tqdm_desc,
    )


def translate_tqa_with_fn(
    tqa: Any,
    translate_fn: Callable[[str], str],
    max_workers: int = 10,
    show_progress: bool = True,
    tqdm_desc: str = "Translating",
):
    """
    Translate the TQA dataset to the target style using GPT4.
    """
    tqa = load_dataset("truthful_qa", "multiple_choice")
    tqa_strings = collect_all_tqa_strings(tqa)
    translations = translate_strings_parallel(
        tqa_strings,
        translate_fn=translate_fn,
        max_workers=max_workers,
        show_progress=show_progress,
        tqdm_desc=tqdm_desc,
    )

    def translate_row(tqa_row: dict[str, Any]) -> dict[str, Any]:
        tqa_row["question"] = translations[tqa_row["question"]]
        for target in ["mc1_targets", "mc2_targets"]:
            for i, choice in enumerate(tqa_row[target]["choices"]):
                tqa_row[target]["choices"][i] = translations[choice]
        return tqa_row

    return tqa.map(translate_row)


def translate_tqa_to_style(
    tqa: Any, style: str, max_workers: int = 10, show_progress: bool = True
):
    """
    Translate the TQA dataset to the target style using GPT4.
    """
    translate_fn = partial(gpt4_style_translate, style=style)
    return translate_tqa_with_fn(
        tqa,
        translate_fn,
        max_workers=max_workers,
        show_progress=show_progress,
        tqdm_desc=f"Translating to {style}",
    )


def translate_tqa_to_language(
    tqa: Any, language: str, max_workers: int = 10, show_progress: bool = True
):
    """
    Translate the TQA dataset to the target language using GPT4.
    """
    translate_fn = partial(gpt4_language_translate, language=language)
    return translate_tqa_with_fn(
        tqa,
        translate_fn,
        max_workers=max_workers,
        show_progress=show_progress,
        tqdm_desc=f"Translating to {language}",
    )


def fs_norm(text: str) -> str:
    """
    Simplify the text for use in a filesystem path.
    """
    return re.sub(r"\s+", "_", text.lower())


def main() -> None:
    tqa = load_dataset("truthful_qa", "multiple_choice")
    translate_dir = Path(Environ.TranslatedDatasetsDir)
    for style in TARGET_STYLES:
        file = translate_dir / f"truthful_qa_{fs_norm(style)}.jsonl"
        if file.exists():
            print(f"File {file} already exists, skipping.")
            continue
        translated_tqa = translate_tqa_to_style(tqa, style, max_workers=100)
        # can't save the outer DatasetDict object using to_json, so just save inner dataset
        translated_tqa["validation"].to_json(file)
    for language in TARGET_LANGUAGES:
        translated_tqa = translate_tqa_to_language(tqa, language, max_workers=100)
        # can't save the outer DatasetDict object using to_json, so just save inner dataset
        file = translate_dir / f"truthful_qa_{fs_norm(language)}.jsonl"
        if file.exists():
            print(f"File {file} already exists, skipping.")
            continue
        translated_tqa["validation"].to_json(file)


if __name__ == "__main__":
    main()
