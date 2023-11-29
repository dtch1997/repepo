import abc
from typing import List
from typing_extensions import override

from repepo.core.types import Completion


def completion_to_str(completion: Completion) -> str:
    return completion.prompt.rstrip() + " " + completion.response.lstrip() + "\n"


class AbstractPrompter(abc.ABC):
    """Interface for modifying completions"""

    @abc.abstractmethod
    def apply(self, completion: Completion) -> Completion:
        raise NotImplementedError()

    def apply_list(self, completions: List[Completion]) -> List[Completion]:
        return [self.apply(completion) for completion in completions]


class IdentityPrompter(AbstractPrompter):
    """Return the prompt as-is"""

    @override
    def apply(self, completion):
        return completion


class FewShotPrompter(AbstractPrompter):
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
