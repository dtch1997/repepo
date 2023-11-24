import abc
from collections import namedtuple
from typing import Any, List, NewType

# Placeholder type definitions
Model = NewType("Model", Any)
Tokenizer = NewType("Tokenizer", Any)
Prompter = NewType("Prompter", Any)
Formatter = NewType("Formatter", Any)

# Base types
Example = namedtuple("Example", ("instruction", "input", "output"))
Completion = namedtuple("Completion", ("prompt", "response"))


class BaseDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def instruction(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def examples(self) -> List[Example]:
        pass


class BasePipeline(abc.ABC):
    @property
    @abc.abstractmethod
    def model(self) -> Model:
        pass

    @property
    @abc.abstractmethod
    def tokenizer(self) -> Tokenizer:
        pass

    @property
    @abc.abstractmethod
    def prompter(self) -> Prompter:
        pass

    @property
    @abc.abstractmethod
    def formatter(self) -> Formatter:
        pass
