from typing import Any
from datasets import load_dataset
from repepo.data.utils import collect_all_strings_recursive


def test_collect_all_tqa_strings() -> None:
    tqa: Any = load_dataset("truthful_qa", "multiple_choice")
    tqa_strings = collect_all_strings_recursive(tqa)
    assert len(tqa_strings) == 6157
