import abc
from typing import List

from repepo.core.types import Completion
from repepo.core.types import Example


class AbstractFormatter(abc.ABC):
    """Describes how to format examples as completions"""

    @abc.abstractmethod
    def apply(self, example: Example, **kwargs) -> str:
        raise NotImplementedError()

    def apply_list(self, examples: List[Example]) -> List[Completion]:
        completions = []
        for example in examples:
            completion = self.apply(example)
            completions.append(completion)
        return completions


class InputOutputFormatter(AbstractFormatter):
    """Format as a simple input-output pair."""

    PROMPT_TEMPLATE = "Input: {instruction} {input} \n" "Output: "

    def apply(self, example: Example, **kwargs):
        del kwargs
        return Completion(
            prompt=self.PROMPT_TEMPLATE.format(
                instruction=example.instruction, input=example.input
            ),
            response=example.output,
        )


class InstructionFormatter(AbstractFormatter):
    """Instruction formatter used for fine-tuning Alpaca."""

    PROMPT_INPUT: str = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
    PROMPT_NO_INPUT: str = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

    def apply(self, example: Example, **kwargs):
        del kwargs
        if bool(example.input):
            prompt = self.PROMPT_INPUT.format(
                instruction=example.instruction, input=example.input
            )
        else:
            prompt = self.PROMPT_NO_INPUT.format_map(
                {"instruction": example.instruction}
            )
        response = example.output
        return Completion(prompt=prompt, response=response)
