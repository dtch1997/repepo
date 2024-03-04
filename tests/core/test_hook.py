from repepo.core.types import Tokenizer
from repepo.core.hook import (
    _find_generation_start_token_index,
)

# TODO: Add tests for the SteeringHook class


def test_find_generation_start_token_index_with_trailing_space(
    tokenizer: Tokenizer,
) -> None:
    base_prompt = "Paris is in: "
    full_prompt = "Paris is in: France"
    assert _find_generation_start_token_index(tokenizer, base_prompt, full_prompt) == 3


def test_find_generation_start_token_index_with_trailing_special_chars(
    tokenizer: Tokenizer,
) -> None:
    base_prompt = "<s> Paris is in: </s>"
    full_prompt = "<s> Paris is in: France </s>"
    assert _find_generation_start_token_index(tokenizer, base_prompt, full_prompt) == 6


def test_find_generation_start_token_base(
    tokenizer: Tokenizer,
) -> None:
    base_prompt = "<s> Paris is in:"
    full_prompt = "<s> Paris is in: France"
    assert _find_generation_start_token_index(tokenizer, base_prompt, full_prompt) == 6
