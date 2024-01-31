from repepo.core.types import Example
from repepo.data.multiple_choice.make_truthfulqa import convert_hf_truthfulqa_dataset


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
