from dataclasses import dataclass
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


Dataset = List[Example]


@dataclass
class Pipeline:
    model: Model
    tokenizer: Tokenizer
    prompter: Prompter
    formatter: Formatter
