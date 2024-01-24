from textwrap import dedent
from transformers import GPTNeoXForCausalLM
from syrupy import SnapshotAssertion

from repepo.core.pipeline import Pipeline, _build_full_prompt
from repepo.core.format import LlamaChatFormatter
from repepo.core.types import Example, Tokenizer


def test_basic_Pipeline_generate(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.generate(
        Example(instruction="Respond", input="A B C D", output="E"),
        generation_config=None,
    )
    # pythia-70m generates nonsense, so just verify we get something
    assert isinstance(res, str)
    assert len(res) > 0


def test_basic_Pipeline_build_generation_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.build_generation_prompt(
        Example(instruction="Respond", input="A B C D", output="E")
    )
    assert res == "Input: Respond A B C D \nOutput:"


def test_icl_Pipeline_build_generation_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer, snapshot: SnapshotAssertion
) -> None:
    dataset = [
        Example(instruction="", input="Paris is in", output="France"),
        Example(instruction="", input="London is in", output="England"),
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]
    pipeline = Pipeline(model, tokenizer)
    pipeline.conversation_wrapper.conversation_history = dataset
    res = pipeline.build_generation_prompt(
        Example(instruction="", input="Beijing is in", output="China"),
    )
    assert res == snapshot


def test_basic_pipeline_calculate_output_logprobs(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.calculate_output_logprobs(
        Example(instruction="Select the best answer.", input="A B C D", output="D")
    )
    assert res.sum_logprobs < 0
    assert res.text == "Input: Select the best answer. A B C D \nOutput: D"
    assert (
        "".join([tok.text for tok in res.token_probs])
        # "Input" is the first token, so the model doesn't predict this
        == ": Select the best answer. A B C D \nOutput: D"
    )
    for tok in res.token_probs:
        assert tok.logprob < 0


def test_pipeline_can_reproduce_caa_train_promts(
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
    example = Example(
        instruction="",
        input=dedent(
            """
            Hello, my name is Linda Johnson ...

            Choices:
            (A) Agree
            (B) Disagree
            """
        ),
        output="(B)",
    )
    completion = pipeline.build_completion(example)
    assert _build_full_prompt(completion).strip() == target.strip()
