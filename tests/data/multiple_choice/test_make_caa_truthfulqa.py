from repepo.core.types import Example, Completion
from repepo.data.multiple_choice.make_caa_truthfulqa import (
    convert_hf_truthfulqa_caa_dataset,
)


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
            positive=Completion(
                prompt="What bear is best?\n\n(A) brown bear\n(B) black bear",
                response="(B)",
            ),
            negative=Completion(
                prompt="What bear is best?\n\n(A) brown bear\n(B) black bear",
                response="(A)",
            ),
        ),
        Example(
            positive=Completion(
                prompt="What is the best color?\n\n(A) blue\n(B) red", response="(A)"
            ),
            negative=Completion(
                prompt="What is the best color?\n\n(A) blue\n(B) red", response="(B)"
            ),
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
            positive=Completion(
                prompt="What bear is best?\n\n(A) brown bear\n(B) black bear",
                response="(B)",
            ),
            negative=Completion(
                prompt="What bear is best?\n\n(A) brown bear\n(B) black bear",
                response="(A)",
            ),
        ),
        Example(
            positive=Completion(
                prompt="What bear is best?\n\n(A) black bear\n(B) panda bear",
                response="(A)",
            ),
            negative=Completion(
                prompt="What bear is best?\n\n(A) black bear\n(B) panda bear",
                response="(B)",
            ),
        ),
        Example(
            positive=Completion(
                prompt="What is the best color?\n\n(A) red\n(B) blue", response="(B)"
            ),
            negative=Completion(
                prompt="What is the best color?\n\n(A) red\n(B) blue", response="(A)"
            ),
        ),
    ]
