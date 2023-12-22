from typing import Any, List, Optional

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
    incorrect_outputs: Optional[list[str]] = None
    meta: Optional[dict[str, Any]] = None


@dataclass
class Completion:
    prompt: str
    response: str


Dataset = List[Example]
