import abc
from collections import namedtuple
from typing import List, Dict, Any, NewType

# Placeholder type definitions
Model = NewType('Model', Any)
Tokenizer = NewType('Tokenizer', Any)
Prompter = NewType('Prompter', Any)
Formatter = NewType('Formatter', Any)

# Base types
Example = namedtuple('Example', ('instruction', 'input', 'output'))
Completion = namedtuple('Completion', ('prompt', 'response'))

class BaseDataset(abc.ABC):

    @property
    def instruction(self) -> str:
        return self._instruction 
    
    @property
    def examples(self) -> List[Example]:
        return self._examples

class BasePipeline(abc.ABC):

    @property
    def model(self) -> Model:
        return self._model
    
    @property 
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer
    
    @property 
    def prompter(self) -> Prompter: 
        return self._prompter 
    
    @property
    def formatter(self) -> Formatter: 
        return self._formatter