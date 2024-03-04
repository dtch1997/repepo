from repepo.core.types import Example, Completion
from repepo.data.multiple_choice.make_caa_sycophancy import make_ab_prompt


def test_make_ab_prompt():
    raw_example = {
        "question": "Hello, my name is John Doe. (A) I require food. (B) I do not require food.",
        "answer_matching_behavior": "(A)",
        "answer_not_matching_behavior": "(B)",
    }
    ab_example = make_ab_prompt(raw_example)
    assert ab_example == Example(
        positive=Completion(
            prompt="Hello, my name is John Doe. (A) I require food. (B) I do not require food.",
            response="(A)",
        ),
        negative=Completion(
            prompt="Hello, my name is John Doe. (A) I require food. (B) I do not require food.",
            response="(B)",
        ),
        steering_token_index=-2,
    )
