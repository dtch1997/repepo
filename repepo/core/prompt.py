import random
import abc

from typing import List
from repepo.core.types import Completion

def completion_to_str(completion: Completion) -> str:
    return completion['prompt'] + completion['response']

class AbstractPrompter(abc.ABC):
    """ Interface for modifying completions """
        
    @abc.abstractmethod
    def apply(self, completion: Completion, **kwargs) -> Completion:
        raise NotImplementedError()

    def apply_list(self, completions: List[Completion]) -> List[Completion]:
        completions_out = []
        for completion in completions:
            completion_out = self.apply(completion)
            completions_out.append(completion_out)
        return completions_out

class IdentityPrompter(AbstractPrompter):
    """ Return the prompt as-is """
    def apply(self, completion, **kwargs):
        del kwargs
        return completion

class FewShotPrompter(AbstractPrompter):
    """ Compose examples few-shot """
    
    def __init__(self, n_few_shot_examples: int = 2):
        self.n_few_shot_examples = n_few_shot_examples

    def apply(self, completion, few_shot_examples=[]):
        prompt = '\n'.join(few_shot_examples) + '\n' + completion['prompt']
        response = completion['response']
        return Completion(
            prompt=prompt, 
            response=response
        )
    
    def apply_list(self, completions: List[Completion]) -> List[Completion]:

        output_completions = []

        for i, completion in enumerate(completions):

            # Sample different completions for context
            few_shot_examples = []
            selected_idxes = [i,]        
            for _ in range(self.n_few_shot_examples):
                done = False
                while not done:
                    idx = random.randint(0, len(completions) - 1) # range inclusive
                    done = idx not in selected_idxes
                few_shot_examples.append(
                    completion_to_str(completions[idx])
                )
                selected_idxes.append(idx)

            # Concatenate completions
            output_completion = self.apply(
                completion, 
                few_shot_examples=few_shot_examples
            )
            output_completions.append(output_completion)

        return output_completions