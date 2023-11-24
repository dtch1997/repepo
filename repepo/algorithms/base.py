import abc

from repepo.core import Dataset, Pipeline


class BaseAlgorithm(abc.ABC):
    @abc.abstractmethod
    def run(self, pipeline: Pipeline, dataset: Dataset) -> Pipeline:
        raise NotImplementedError()


if __name__ == "__main__":
    # Construct pipeline
    # Evaluate pipeline before
    # Run algorithm
    # Evaluate pipeline after
    pass
