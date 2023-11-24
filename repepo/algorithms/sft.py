from repepo.core import Dataset
from repepo.core import Pipeline

from .base import BaseAlgorithm


class SupervisedFineTuning(BaseAlgorithm):
    def run(self, pipeline: Pipeline, dataset: Dataset) -> Pipeline:
        """Modifies the base model weights"""

        # Make supervised data module
        # Run training, with optional WandB eval

        # keep pyright happy for now
        return pipeline
