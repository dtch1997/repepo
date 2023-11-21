from repepo.core import BaseDataset
from repepo.core import BasePipeline

from .base import BaseAlgorithm


class SupervisedFineTuning(BaseAlgorithm):
    def run(self, pipeline: BasePipeline, dataset: BaseDataset) -> BasePipeline:
        """Modifies the base model weights"""

        # Make supervised data module
        # Run training, with optional WandB eval
