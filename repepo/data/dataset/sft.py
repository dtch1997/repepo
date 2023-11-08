""" Dataset for supervised fine-tuning workflow. """
import copy
import torch
import transformers
import logging

from torch.utils.data import Dataset
from typing import Dict, Sequence, Any
from dataclasses import dataclass
from repepo.data.utils import tokenize, IGNORE_INDEX

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    padding: Any = None,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized, targets_tokenized = [tokenize(strings, tokenizer, padding) for strings in (examples, sources, targets)]
    input_ids = examples_tokenized["input_ids"]
    
    # Set labels to IGNORE_INDEX for prompt tokens
    # This ensures that loss is not backprop'd to prompt tokens
    labels = copy.deepcopy(input_ids)
    for label, source_ids in zip(labels, sources_tokenized["input_ids"]):
        source_len = len(source_ids)
        label[:source_len] = IGNORE_INDEX
    return dict(
        input_ids=input_ids, 
        labels=labels, 
        prompt_ids = sources_tokenized['input_ids'],
        reference_ids = targets_tokenized['input_ids']
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict, tokenizer: transformers.PreTrainedTokenizer, padding: Any = None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        # Map the examples into prompt format 
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        # Add explicit EOS token
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, padding)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.prompt_ids = data_dict["prompt_ids"]
        self.reference_ids = data_dict["reference_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            prompt_ids=self.prompt_ids[i],
            reference_ids=self.reference_ids[i]
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        keys = instances[0].keys()
        return_dict = {}

        # Prepare inputs for model training
        input_ids, labels = tuple(
            [instance[key] for instance in instances] \
                for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return_dict['input_ids'] = input_ids
        return_dict['labels'] = labels
        return_dict['attention_mask'] = input_ids.ne(self.tokenizer.pad_token_id)

        # Optionally, prepare inputs for model eval
        # TODO: figure out how to do this more nicely...
        if 'prompt_ids' in keys:
            prompt_ids = [instance['prompt_ids'] for instance in instances]
            prompt_ids = torch.nn.utils.rnn.pad_sequence(
                prompt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        if 'reference_ids' in keys:
            reference_ids = [instance['reference_ids'] for instance in instances]
            reference_ids = torch.nn.utils.rnn.pad_sequence(
                reference_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        
        return return_dict
