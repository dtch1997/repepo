import abc
from typing import Any, List, NewType, NamedTuple

# Placeholder type definitions
Model = NewType("Model", Any)
Tokenizer = NewType("Tokenizer", Any)
Prompter = NewType("Prompter", Any)
Formatter = NewType("Formatter", Any)


# Base types
class Example(NamedTuple):
    instruction: str
    input: str
    output: str


class Completion(NamedTuple):
    prompt: str
    response: str


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
