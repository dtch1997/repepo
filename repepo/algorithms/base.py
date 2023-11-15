import abc 
from repepo.core import BaseDataset, BasePipeline

class BaseAlgorithm(abc.ABC):
    
    @abc.abstractmethod
    def run(self, pipeline: BasePipeline, dataset: BaseDataset) -> BasePipeline:
        raise NotImplementedError()
    

if __name__ == "__main__": 
    # Construct pipeline
    # Evaluate pipeline before
    # Run algorithm
    # Evaluate pipeline after
    pass