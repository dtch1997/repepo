from .base import BaseAlgorithm
from repepo.core import BaseDataset, BasePipeline

class SupervisedFineTuning(BaseAlgorithm):

    def run(self, pipeline: BasePipeline, dataset: BaseDataset) -> BasePipeline:
        """ Modifies the base model weights """

        # Make supervised data module
        # Run training, with optional WandB eval