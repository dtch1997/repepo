from typing import Any

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

# Placeholder type definitions
Model = PreTrainedModel
Tokenizer = PreTrainedTokenizerBase


# Base types
@dataclass
class Completion:
    prompt: str
    response: str


@dataclass
class Example:
    positive: Completion
    negative: Completion
    meta: dict[str, Any] | None = None
    steering_token_index: int = -1  # Token index to extract and apply steering vectors


Dataset = list[Example]
