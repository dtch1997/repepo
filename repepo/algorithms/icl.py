from dataclasses import replace
from repepo.core.prompt import FewShotPrompter
from .base import Algorithm
from repepo.core import Pipeline, Dataset


class InContextLearning(Algorithm):
    def run(self, pipeline: Pipeline, dataset: Dataset) -> Pipeline:
        """Uses an in-context learning prefix to prompts"""

        icl_completions = pipeline.formatter.apply_list(dataset)
        new_prompter = FewShotPrompter(icl_completions)

        return replace(pipeline, prompter=new_prompter)
