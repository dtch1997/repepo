from pathlib import Path
from typing import Any, Callable
from datasets import load_dataset
from repepo.data.utils import translate_row_recursive, collect_all_strings_recursive
from repepo.translation.constants import LangOrStyleCode, LANG_OR_STYLE_MAPPING
from repepo.translation.lang_or_style_translate import lang_or_style_translate
from repepo.variables import Environ


def translate_tqa(
    tqa: Any,
    lang_or_style: LangOrStyleCode,
    max_workers: int = 10,
    show_progress: bool = True,
):
    """
    Translate the TQA dataset to the target style using GPT4.
    """
    tqa_strings = collect_all_strings_recursive(tqa)
    translations = lang_or_style_translate(
        tqa_strings,
        lang_or_style,
        gpt_worker_threads=max_workers,
        show_progress=show_progress,
    )
    return tqa.map(lambda row: translate_row_recursive(row, translations))


def run_and_save(run_fn: Callable[[], Any], file: Path, force: bool = False) -> None:
    """
    Save the translated TQA dataset to a file.
    """
    if file.exists() and not force:
        print(f"File {file} already exists, skipping.")
        return
    tqa = run_fn()
    tqa["validation"].to_json(file, force_ascii=False)


def main() -> None:
    tqa = load_dataset("truthful_qa", "multiple_choice")
    translate_dir = Path(Environ.TranslatedDatasetsDir)
    for code in LANG_OR_STYLE_MAPPING.keys():
        file = translate_dir / code / "truthful_qa.jsonl"
        if file.exists():
            print(f"File {file} already exists, skipping.")
            continue
        print(f"Translating TQA to {code}")
        translated_tqa = translate_tqa(tqa, code, max_workers=100)
        # can't save DatasetDict to json, so just save the validation split (there's only 1 split anyway)
        translated_tqa["validation"].to_json(file, force_ascii=False)


if __name__ == "__main__":
    main()
