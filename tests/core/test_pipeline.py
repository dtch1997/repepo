from textwrap import dedent
from transformers import GPTNeoXForCausalLM

from repepo.core.pipeline import Pipeline
from repepo.core.format import IdentityFormatter, LlamaChatFormatter
from repepo.core.types import Completion, Tokenizer
from syrupy.assertion import SnapshotAssertion


def test_basic_Pipeline_build_generation_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.build_generation_prompt(
        Completion(prompt="Respond A B C D", response="E")
    )
    assert res == "Respond A B C D"


def test_basic_Pipeline_build_full_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.build_full_prompt(Completion(prompt="Respond A B C D", response="E"))
    assert res == "Respond A B C D E"


def test_basic_Pipeline_build_generation_prompt_with_custom_completion_template(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    formatter = IdentityFormatter(
        completion_template="Input: {prompt}\n Output: My answer is: {response}",
    )
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    res = pipeline.build_generation_prompt(
        Completion(prompt="Respond A B C D", response="E")
    )
    assert res == "Input: Respond A B C D\n Output: My answer is:"


def test_basic_Pipeline_build_full_prompt_with_custom_completion_template(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    formatter = IdentityFormatter(
        completion_template="Input: {prompt}\n Output: My answer is: {response}",
    )
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    res = pipeline.build_full_prompt(
        Completion(prompt="Respond A B C D", response="E"),
    )
    assert res == "Input: Respond A B C D\n Output: My answer is: E"


def test_icl_Pipeline_build_generation_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer, snapshot: SnapshotAssertion
) -> None:
    dataset = [
        Completion(prompt="Paris is in", response="France"),
        Completion(prompt="London is in", response="England"),
        Completion(prompt="Berlin is in", response="Germany"),
    ]
    pipeline = Pipeline(model, tokenizer)
    pipeline.conversation_history = dataset
    res = pipeline.build_generation_prompt(
        Completion(prompt="Beijing is in", response="China")
    )
    assert res == snapshot


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

def test_pipeline_can_reproduce_caa_train_prompts(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    # copied from a real prompt in the CAA training set, abbreviated for brevity
    target = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>
        
        Hello, my name is Linda Johnson ...
        Choices:
        (A) Agree
        (B) Disagree [/INST] (B)
        """
    )
    formatter = LlamaChatFormatter()
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    completion = Completion(
        prompt=dedent(
            """
            Hello, my name is Linda Johnson ...
            Choices:
            (A) Agree
            (B) Disagree
            """
        ),
        response = "(B)",
    )
    assert pipeline.build_full_prompt(completion).strip() == target.strip()
