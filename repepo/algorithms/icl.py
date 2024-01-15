from repepo.core.prompt import FewShotPrompter
from .base import Algorithm
from repepo.core import Pipeline, Dataset
from typing import Any


class InContextLearning(Algorithm):
    max_icl_examples: int

    def __init__(self, max_icl_examples: int = 5):
        super().__init__()
        self.max_icl_examples = max_icl_examples

    def run(self, pipeline: Pipeline, dataset: Dataset) -> dict[str, Any]:
        """Uses an in-context learning prefix to prompts"""

        icl_completions = pipeline.formatter.apply_list(
            dataset[: self.max_icl_examples]
        )
        new_prompter = FewShotPrompter(icl_completions)
        pipeline.prompter = new_prompter

        return {}
