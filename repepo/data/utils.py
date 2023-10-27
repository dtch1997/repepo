import transformers
import copy
import torch

from typing import Dict, Sequence, NewType

SpecialToken = NewType("SpecialToken", str)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_pad_token(tokenizer: transformers.AutoTokenizer) -> Dict[str, SpecialToken]:
    """ Set the padding token of the tokenizer if it doesn't have one 
    
    Handled separately from other tokens because this can be used for pure inference workflows. 
    I.e. we'll never backpropagate loss to the padding token, so it's safe to use an arbitrary value.
    """
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    return special_tokens_dict

def get_special_tokens(tokenizer: transformers.AutoTokenizer) -> Dict[str, SpecialToken]:
    """ Get special tokens that must be added to model, tokenizer """    
    # Add special tokens
    special_tokens_dict = dict()

    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    return special_tokens_dict

def tokenize(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict[str, torch.Tensor]:
    """ Tokenize a list of strings.
    
    Does not do any padding. 
    Truncates the sequences to the model's max length.
    Returns tensors of input_ids and labels
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
    )