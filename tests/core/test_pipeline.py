from transformers import GPTNeoXForCausalLM

from repepo.core.pipeline import Pipeline
from repepo.core.types import Completion, Tokenizer


def test_basic_pipeline_calculate_output_logprobs(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.calculate_output_logprobs(
        Completion(
            prompt="Input: Select the best answer.\nA B C D", response="Output: D"
        )
    )
    assert res.sum_logprobs < 0
    assert res.text == "Input: Select the best answer.\nA B C D Output: D"
    assert (
        "".join([tok.text for tok in res.token_probs])
        # "Input" is the first token, so the model doesn't predict this
        == ": Select the best answer.\nA B C D Output: D"
    )
    for tok in res.token_probs:
        assert tok.logprob < 0
