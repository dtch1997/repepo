from repepo.core.types import Example
from repepo.data.multiple_choice.make_truthfulqa import (
    convert_hf_truthfulqa_dataset,
    convert_hf_truthfulqa_caa_dataset,
)


def test_convert_hf_truthfulqa_dataset() -> None:
    hf_dataset = {
        "question": ["q1", "q2"],
        "mc1_targets": [
            {"labels": [0, 1, 0], "choices": ["a", "b", "c"]},
            {"labels": [1, 0], "choices": ["c", "d"]},
        ],
    }
    tqa_dataset = convert_hf_truthfulqa_dataset(hf_dataset)
    assert tqa_dataset == [
        Example(
            instruction="",
            input="q1",
            output="b",
            incorrect_outputs=["a", "c"],
        ),
        Example(
            instruction="",
            input="q2",
            output="c",
            incorrect_outputs=["d"],
        ),
    ]


def test_convert_hf_truthfulqa_caa_dataset_picks_first_incorrect_by_default() -> None:
    hf_dataset = {
        "question": ["What bear is best?", "What is the best color?"],
        "mc1_targets": [
            {
                "labels": [0, 1, 0],
                "choices": ["brown bear", "black bear", "panda bear"],
            },
            {"labels": [1, 0], "choices": ["blue", "red"]},
        ],
    }
    tqa_dataset = convert_hf_truthfulqa_caa_dataset(hf_dataset)
    assert tqa_dataset == [
        Example(
            instruction="",
            input="What bear is best?\n\n(A) brown bear\n(B) black bear",
            output="(B)",
            incorrect_outputs=["(A)"],
        ),
        Example(
            instruction="",
            input="What is the best color?\n\n(A) blue\n(B) red",
            output="(A)",
            incorrect_outputs=["(B)"],
        ),
    ]


def test_convert_hf_truthfulqa_caa_dataset_can_duplicate_correct_answers() -> None:
    hf_dataset = {
        "question": ["What bear is best?", "What is the best color?"],
        "mc1_targets": [
            {
                "labels": [0, 1, 0],
                "choices": ["brown bear", "black bear", "panda bear"],
            },
            {"labels": [1, 0], "choices": ["blue", "red"]},
        ],
    }
    tqa_dataset = convert_hf_truthfulqa_caa_dataset(
        hf_dataset, multi_answer_method="repeat_correct"
    )
    print(tqa_dataset)
    assert tqa_dataset == [
        Example(
            instruction="",
            input="What bear is best?\n\n(A) brown bear\n(B) black bear",
            output="(B)",
            incorrect_outputs=["(A)"],
        ),
        Example(
            instruction="",
            input="What bear is best?\n\n(A) black bear\n(B) panda bear",
            output="(A)",
            incorrect_outputs=["(B)"],
        ),
        Example(
            instruction="",
            input="What is the best color?\n\n(A) red\n(B) blue",
            output="(B)",
            incorrect_outputs=["(A)"],
        ),
    ]
