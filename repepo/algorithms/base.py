import abc

from repepo.core import Dataset, Pipeline
from typing import Any


class Algorithm(abc.ABC):
    @abc.abstractmethod
    def run(
        self, pipeline: Pipeline, dataset: Dataset, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError()


if __name__ == "__main__":
    # Construct pipeline
    # Evaluate pipeline before
    # Run algorithm
    # Evaluate pipeline after
    pass
