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

    # TODO(dtch1997): There has to be some way to use metaclasses
    # to put the from_dict, to_dict methods in both Example, Completion.
    @classmethod
    def from_dict(cls, data_dict):
        return cls(**data_dict)

    def to_dict(self):
        return self._asdict()


class Completion(NamedTuple):
    prompt: str
    response: str

    @classmethod
    def from_dict(cls, data_dict):
        return cls(**data_dict)

    def to_dict(self):
        return self._asdict()


Dataset = List[Example]
