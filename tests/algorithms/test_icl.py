from transformers import GPTNeoXForCausalLM
from repepo.algorithms.icl import InContextLearning
from repepo.core.pipeline import Pipeline
from repepo.core.prompt import FewShotPrompter

from repepo.core.types import Completion, Example, Tokenizer


def test_InContextLearning_run(model: GPTNeoXForCausalLM, tokenizer: Tokenizer) -> None:
    pipeline = Pipeline(model, tokenizer)
    algorithm = InContextLearning()
    dataset = [
        Example(instruction="", input="Paris is in", output="France"),
        Example(instruction="", input="London is in", output="England"),
        Example(instruction="", input="Berlin is in", output="Germany"),
    ]

    algorithm.run(pipeline, dataset=dataset)

    assert isinstance(pipeline.prompter, FewShotPrompter)
    assert pipeline.prompter.few_shot_completions == [
        Completion(prompt="Input:  Paris is in \nOutput: ", response="France"),
        Completion(prompt="Input:  London is in \nOutput: ", response="England"),
        Completion(prompt="Input:  Berlin is in \nOutput: ", response="Germany"),
    ]
