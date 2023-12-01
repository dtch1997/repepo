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


class RepExample(NamedTuple):
    instruction: str 
    input: str
    output: str # for repe, we do not need an ouput, but this is not necessarily going to be the same as direction. Eg. we could have an example for 'the amount of bias is' and the output could be 'high' or 'low', not True or False
    id: int # for repe reading, we have two examples per id, one for each direction
    direction: bool # True for positive, False for negative


class Completion(NamedTuple):
    prompt: str
    response: str


Dataset = Union[List[Example], List[RepExample]]
