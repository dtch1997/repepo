from typing import List

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

# Placeholder type definitions
Model = PreTrainedModel
Tokenizer = PreTrainedTokenizerBase


# Base types
@dataclass
class Example:
    instruction: str
    input: str
    output: str


@dataclass
class Completion:
    prompt: str
    response: str


Dataset = List[Example]
