import abc
from dataclasses import dataclass
from typing import List
from typing_extensions import override

from repepo.core.types import Completion

LLAMA_7B_DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
LLAMA_7B_DEFAULT_COMPLETION_TEMPLATE = "{prompt} {response}"


@dataclass
class FormatContext:
    """
    Context provided to the formatter in the format() method.
    """

    index: int
    completions: List[Completion]

    @property
    def num_completions(self) -> int:
        return len(self.completions)


class Formatter(abc.ABC):
    """Describes how to format examples as completions"""

    msg_separator: str = "\n"
    system_prompt: str = ""
    completion_template: str = "{prompt} {response}"

    def __init__(
        self,
        completion_template: str = "{prompt} {response}",
        msg_separator: str = "\n",
    ) -> None:
        self.msg_separator = msg_separator
        self.completion_template = completion_template

    @abc.abstractmethod
    def format(self, completion: Completion, ctx: FormatContext) -> Completion:
        """
        Format a completion as another completion. Subclasses should override this method.
        This method should not be called directly externally, instead use format_conversation().
        """
        pass

    @property
    def prompt_only_completion_template(self) -> str:
        return self.completion_template.replace("{response}", "").strip()

    def format_prompt_as_str(self, completion: Completion) -> str:
        """
        Format a completion's prompt as a string.
        """
        return self.prompt_only_completion_template.format(
            prompt=completion.prompt.strip()
        )

    def format_as_str(self, completion: Completion) -> str:
        """
        Format a completion as a string.
        """
        return self.completion_template.format(
            prompt=completion.prompt.strip(), response=completion.response.strip()
        )

    def format_conversation(
        self,
        current_message: Completion,
        history: list[Completion] = [],
    ) -> Completion:
        """
        Generate a completion for a conversation, handling ICL convo history
        """
        conversation = [*history, current_message]
        completions: list[Completion] = []
        for i, completion in enumerate(conversation):
            ctx = FormatContext(index=i, completions=conversation)
            completion = self.format(completion, ctx)
            completions.append(completion)
        prefix_completions = completions[:-1]
        final_completion = completions[-1]
        convo_prefix = self.msg_separator.join(
            self.format_as_str(completion) for completion in prefix_completions
        )
        prompt = final_completion.prompt.strip()
        if len(convo_prefix) > 0:
            prompt = convo_prefix + self.msg_separator + final_completion.prompt
        return Completion(prompt=prompt, response=final_completion.response)


class IdentityFormatter(Formatter):
    """Do nothing."""

    @override
    def format(self, completion: Completion, ctx: FormatContext):
        return completion


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
        completion_template: str = "{prompt} {response}",
        msg_separator: str = "\n",
        system_prompt: str | None = "You are a helpful, honest and concise assistant.",
    ) -> None:
        self.system_prompt = system_prompt
        super().__init__(
            completion_template=completion_template, msg_separator=msg_separator
        )

    @override
    def format(self, completion: Completion, ctx: FormatContext):
        dialog_content_parts = []
        # If first example, add system prompt
        if ctx.index == 0 and self.system_prompt is not None:
            dialog_content_parts.append(f"{self.B_SYS}{self.system_prompt}{self.E_SYS}")
        dialog_content_parts.append(completion.prompt.strip())
        dialog_content = "\n".join(dialog_content_parts)

        # Add [INST] and [/INST] tags
        prompt = f"{self.B_INST} {dialog_content} {self.E_INST} "
        response = completion.response.strip()
        return Completion(prompt=prompt, response=response)
