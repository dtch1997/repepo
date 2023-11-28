from typing import List, NamedTuple

from transformers import PreTrainedTokenizerBase, PreTrainedModel

# Placeholder type definitions
Model = PreTrainedModel
Tokenizer = PreTrainedTokenizerBase


# Base types
class Example(NamedTuple):
    instruction: str
    input: str
    output: str


class Completion(NamedTuple):
    prompt: str
    response: str


Dataset = List[Example]
