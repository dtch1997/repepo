import abc

from repepo.core import Dataset, Pipeline
from typing import Dict, Any


class Algorithm(abc.ABC):
    @abc.abstractmethod
    def run(
        self, pipeline: Pipeline, dataset: Dataset, **kwargs: Dict[str, Any]
    ) -> Pipeline:
        raise NotImplementedError()


if __name__ == "__main__":
    # Construct pipeline
    # Evaluate pipeline before
    # Run algorithm
    # Evaluate pipeline after
    pass
