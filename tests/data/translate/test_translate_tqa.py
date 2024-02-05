from typing import Any
from datasets import load_dataset
from repepo.data.translate.translate_tqa import (
    collect_all_tqa_strings,
    translate_tqa_with_fn,
)


def test_collect_all_tqa_strings() -> None:
    tqa: Any = load_dataset("truthful_qa", "multiple_choice")
    tqa_strings = collect_all_tqa_strings(tqa)
    assert len(tqa_strings) == 6157


def test_translate_tqa_with_fn() -> None:
    tqa: Any = load_dataset("truthful_qa", "multiple_choice")

    def translate_fn(orig):
        return f"TRANS: {orig}"

    translated_tqa = translate_tqa_with_fn(tqa, translate_fn)
    assert list(translated_tqa.keys()) == ["validation"]
    assert len(translated_tqa["validation"]) == 817

    for ex, translated_ex in zip(tqa["validation"], translated_tqa["validation"]):
        assert translated_ex == {
            "question": f"TRANS: {ex['question']}",
            "mc1_targets": {
                "choices": [
                    translate_fn(choice) for choice in ex["mc1_targets"]["choices"]
                ],
                "labels": ex["mc1_targets"]["labels"],
            },
            "mc2_targets": {
                "choices": [
                    translate_fn(choice) for choice in ex["mc2_targets"]["choices"]
                ],
                "labels": ex["mc2_targets"]["labels"],
            },
        }
