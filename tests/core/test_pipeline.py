from transformers import GPTNeoXForCausalLM
from syrupy import SnapshotAssertion

from repepo.core.format import InputOutputFormatter
from repepo.core.pipeline import Pipeline
from repepo.core.prompt import FewShotPrompter

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
    assert res == ("Input: Respond A B C D \nOutput: ")


def test_icl_Pipeline_build_generation_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer, snapshot: SnapshotAssertion
) -> None:
    dataset = [
        Example(instruction="", input="Paris is in", output="France"),
        Example(instruction="", input="London is in", output="England"),
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]
    formatter = InputOutputFormatter()
    icl_prompter = FewShotPrompter(formatter.apply_list(dataset))

    pipeline = Pipeline(model, tokenizer, prompter=icl_prompter, formatter=formatter)
    res = pipeline.build_generation_prompt(
        Example(instruction="", input="Beijing is in", output="China"),
    )
    assert res == snapshot
