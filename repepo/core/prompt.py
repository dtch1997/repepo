import abc
from dataclasses import dataclass
from typing import List
from typing_extensions import override

from repepo.core.types import Completion


def completion_to_str(completion: Completion) -> str:
    return completion.prompt.rstrip() + " " + completion.response.lstrip() + "\n"


class Prompter(abc.ABC):
    """Interface for modifying completions"""

    @abc.abstractmethod
    def apply(self, completion: Completion) -> Completion:
        raise NotImplementedError()

    def apply_list(self, completions: List[Completion]) -> List[Completion]:
        return [self.apply(completion) for completion in completions]


class IdentityPrompter(Prompter):
    """Return the prompt as-is"""

    @override
    def apply(self, completion):
        return completion


@dataclass
class LlamaChatPrompter(Prompter):
    """
    Prepend system message before the first prompt. This is based on the SycophancySteering
    prompt format, except the system message is not inside of the [INST] tags. Hopefully
    this is OK?

    TODO: Put the sys message inside of the [INST] tags, or at least make it possible to do so.
          This will require some rearchitecting

    Based on: https://github.com/nrimsky/SycophancySteering/blob/main/utils/tokenize_llama.py#L30
    """

    system_prompt: str = "You are a helpful, honest and concise assistant."

    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"

    @override
    def apply(self, completion: Completion) -> Completion:
        prompt = f"{self.B_SYS}{self.system_prompt}{self.E_SYS}{completion.prompt}"
        response = completion.response
        return Completion(prompt=prompt, response=response)


class FewShotPrompter(Prompter):
    """Compose examples few-shot"""

    # TODO: should these be randomized?
    few_shot_completions: list[Completion]

    def __init__(self, few_shot_completions: list[Completion]):
        self.few_shot_completions = few_shot_completions

    @override
    def apply(self, completion: Completion) -> Completion:
        prompt = ""
        for fs_comp in self.few_shot_completions:
            prompt += completion_to_str(fs_comp) + "\n"
        prompt += completion.prompt

        response = completion.response

        return Completion(prompt=prompt, response=response)
