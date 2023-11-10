import abc 
import random

from typing import Dict, Any, NewType, List

Example = NewType('Example', Dict[str, Any])
# Example: 
#   - a dict with keys 'instruction', 'input', 'output'
#   - could also have additional metadata

Completion = NewType('Completion', Dict[str, Any])
# Completion: 
#   - a dict with keys 'prompt', 'response'

def completion_to_str(completion: Completion) -> str:
    return completion['prompt'] + completion['response']

class AbstractFormatter(abc.ABC):
        
    def format_example(self, example: Example, **kwargs) -> Completion:
        pass

    def apply(self, examples: List[Example]) -> List[Completion]:
        completions = []
        for example in examples:
            completion = self.format_example(example)
            completions.append(completion)
        return completions
    
class QAFormatter(AbstractFormatter):
    """ Format as a simple QA pair. """
    QUESTION = "Q: {instruction} {input}"
    ANSWER = "A: {output}"

    def format_example(self, example: Dict[str, Any], **kwargs):
        del kwargs
        prompt = self.QUESTION.format_map(example)
        response = self.ANSWER.format_map(example)
        return {
            'prompt': prompt, 
            'response': response
        }


class InstructionFormatter(AbstractFormatter):
    """ Instruction formatter used for fine-tuning Alpaca. """

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

    def format_example(self, example: Dict[str, Any], **kwargs):
        del kwargs
        if 'input' in example and bool(example['input']):
            prompt = self.PROMPT_INPUT.format_map(example)
        else:
            prompt = self.PROMPT_NO_INPUT.format_map(example)
        response = example['output']
        return {
            'prompt': prompt, 
            'response': response
        }

class FewShotPrompter:
    """ Compose examples few-shot """
    
    def __init__(self, n_few_shot_examples: int = 2):
        self.n_few_shot_examples = n_few_shot_examples
    
    def apply(self, completions: List[Completion]) -> List[Completion]:

        output_completions = []

        for i, completion in enumerate(completions):

            # Sample different completions for context
            few_shot_examples = []
            selected_idxes = [i, ]        
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
            prompt = '\n'.join(few_shot_examples) + completion['prompt']
            response = completion['response']
            output_completions.append(dict(
                prompt=prompt, 
                response=response
            ))

        return output_completions