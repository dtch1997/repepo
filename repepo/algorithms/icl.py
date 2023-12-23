from repepo.core.prompt import FewShotPrompter
from .base import Algorithm
from repepo.core import Pipeline, Dataset
from typing import Any


class InContextLearning(Algorithm):
    def run(self, pipeline: Pipeline, dataset: Dataset) -> dict[str, Any]:
        """Uses an in-context learning prefix to prompts"""

        icl_completions = pipeline.formatter.apply_list(dataset)
        new_prompter = FewShotPrompter(icl_completions)
        pipeline.prompter = new_prompter

        return {}
