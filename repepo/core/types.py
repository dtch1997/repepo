from typing import List, NamedTuple, Union

from transformers import PreTrainedTokenizerBase, PreTrainedModel
from repepo.repe.rep_control_reading_vec import WrappedReadingVecModel

# Placeholder type definitions
Model = Union[PreTrainedModel, WrappedReadingVecModel]
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
