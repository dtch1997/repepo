from transformers import GPTNeoXForCausalLM
from repepo.algorithms.icl import InContextLearning
from repepo.core.pipeline import Pipeline

from repepo.core.types import Example, Tokenizer


def test_InContextLearning_run(model: GPTNeoXForCausalLM, tokenizer: Tokenizer) -> None:
    pipeline = Pipeline(model, tokenizer)
    algorithm = InContextLearning()
    dataset = [
        Example(instruction="", input="Paris is in", output="France"),
        Example(instruction="", input="London is in", output="England"),
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]

    algorithm.run(pipeline, dataset=dataset)
    assert pipeline.conversation_wrapper.conversation_history == dataset


def test_InContextLearning_run_with_max_icl_examples(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    algorithm = InContextLearning(max_icl_examples=2)
    dataset = [
        Example(instruction="", input="Paris is in", output="France"),
        Example(instruction="", input="London is in", output="England"),
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]

    algorithm.run(pipeline, dataset=dataset)
    assert pipeline.conversation_wrapper.conversation_history == dataset[:2]
