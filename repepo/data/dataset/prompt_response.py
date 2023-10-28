""" Dataset for supervised fine-tuning workflow. """
import copy
import torch
import transformers
import logging

from torch.utils.data import Dataset
from typing import Dict, Sequence, Any, List
from dataclasses import dataclass
from repepo.data.utils import tokenize, IGNORE_INDEX

# Contains 'prompt' and 'response' keys; plus any additional metadata
PromptResponsePair = Dict[str, Any] 

class PromptResponseDataset(Dataset):
    """ Dataset for iterating over prompt-response pairs. """

    def __init__(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor):
        super(PromptResponseDataset, self).__init__()
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids

    def __len__(self):
        return len(self.prompt_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            prompt_ids = self.prompt_ids[i], 
            response_ids = self.response_ids[i]
        )
    