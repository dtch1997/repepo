import abc
from typing import List
from typing_extensions import override

from repepo.core.types import Completion
from repepo.core.types import Example


class Formatter(abc.ABC):
    """Describes how to format examples as completions"""

    @abc.abstractmethod
    def apply(self, example: Example, **kwargs) -> Completion:
        pass

    def apply_list(self, examples: List[Example]) -> List[Completion]:
        completions: list[Completion] = []
        for example in examples:
            completion = self.apply(example)
            completions.append(completion)
        return completions


class InputOutputFormatter(Formatter):
    """Format as a simple input-output pair."""

    PROMPT_TEMPLATE = "Input: {instruction} {input} \n" "Output: "

    @override
    def apply(self, example: Example, **kwargs):
        del kwargs
        return Completion(
            prompt=self.PROMPT_TEMPLATE.format(
                instruction=example.instruction, input=example.input
            ),
            response=example.output,
        )


class LlamaChatFormatter(Formatter):
    """
    Add [INST] and [/INST] tags to the instruction and input.

    Based on: https://github.com/nrimsky/SycophancySteering/blob/main/utils/tokenize_llama.py#L30
    """

    B_INST = "[INST]"
    E_INST = "[/INST]"

    @override
    def apply(self, example: Example):
        dialog_content_parts = []
        if example.instruction:
            dialog_content_parts.append(example.instruction.strip())
        dialog_content_parts.append(example.input.strip())
        dialog_content = "\n".join(dialog_content_parts)
        prompt = f"{self.B_INST} {dialog_content} {self.E_INST} "
        response = example.output.strip()
        return Completion(prompt=prompt, response=response)


class InstructionFormatter(Formatter):
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

    @override
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
