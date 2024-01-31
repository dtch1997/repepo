import abc
from dataclasses import dataclass
from typing import List
from typing_extensions import override

from repepo.core.types import Completion
from repepo.core.types import Example


@dataclass
class FormatContext:
    """
    Context provided to the formatter in the format() method.
    """

    index: int
    examples: List[Example]

    @property
    def num_examples(self) -> int:
        return len(self.examples)


class Formatter(abc.ABC):
    """Describes how to format examples as completions"""

    msg_separator: str = "\n"
    completion_template: str = "{prompt} {response}"

    @abc.abstractmethod
    def format(self, example: Example, ctx: FormatContext) -> Completion:
        """
        Format an example as a completion. Subclasses should override this method.
        This method should not be called directly externally, instead use format_conversation().
        """
        pass

    def format_completion(self, completion: Completion) -> str:
        """
        Format a completion as a string.
        """
        return self.completion_template.format(
            prompt=completion.prompt.strip(), response=completion.response.strip()
        )

    def format_conversation(
        self,
        current_message: Example,
        history: list[Example] = [],
    ) -> Completion:
        """
        Generate a completion for a conversation, handling ICL convo history
        """
        conversation = [*history, current_message]
        completions: list[Completion] = []
        for i, example in enumerate(conversation):
            ctx = FormatContext(index=i, examples=conversation)
            completion = self.format(example, ctx)
            completions.append(completion)
        prefix_completions = completions[:-1]
        final_completion = completions[-1]
        convo_prefix = self.msg_separator.join(
            self.format_completion(completion) for completion in prefix_completions
        )
        prompt = final_completion.prompt.strip()
        if len(convo_prefix) > 0:
            prompt = convo_prefix + self.msg_separator + final_completion.prompt
        return Completion(prompt=prompt, response=final_completion.response)


class InputOutputFormatter(Formatter):
    """Format as a simple input-output pair."""

    PROMPT_TEMPLATE = "Input: {instruction} {input} \n" "Output: "

    def __init__(
        self,
        msg_separator: str = "\n",
        completion_template: str = "{prompt} {response}",
    ) -> None:
        self.completion_template = completion_template
        self.msg_separator = msg_separator

    @override
    def format(self, example: Example, ctx: FormatContext):
        return Completion(
            prompt=self.PROMPT_TEMPLATE.format(
                instruction=example.instruction, input=example.input
            ),
            response=example.output,
        )


class LlamaChatFormatter(Formatter):
    """
    Add [INST] and [/INST] tags to the instruction and input.
    Also adds a system message before the first prompt.

    Based on: https://github.com/nrimsky/SycophancySteering/blob/main/utils/tokenize_llama.py#L30
    """

    system_prompt: str | None

    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n"

    def __init__(
        self,
        system_prompt: str | None = "You are a helpful, honest and concise assistant.",
        msg_separator: str = "\n",
        completion_template: str = "{prompt} {response}",
    ) -> None:
        self.system_prompt = system_prompt
        self.msg_separator = msg_separator
        self.completion_template = completion_template

    @override
    def format(self, example: Example, ctx: FormatContext):
        dialog_content_parts = []
        if ctx.index == 0 and self.system_prompt is not None:
            dialog_content_parts.append(f"{self.B_SYS}{self.system_prompt}{self.E_SYS}")
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
    def format(self, example: Example, ctx: FormatContext):
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
