# from repepo.algorithms.base import Algorithm
import wandb
from typing import List

import torch

from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

# from repepo.utils.logging import WandbConfig, WandbLogger
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from repepo.core.types import Completion


# from repepo.data.dataset import sft

# from repepo.variables import Model

from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset


def prepare_dataset():
    train_dataset = load_dataset("corrigible-neutral-HHH.jsonl")
    return train_dataset


def tokenize_completions(
    completions: List[Completion], tokenizer: AutoTokenizer
) -> List[torch.Tensor]:
    completion_strs: List[str] = [
        f"{completion.prompt} {completion.response}" for completion in completions
    ]
    tokenized_completions = tokenizer(completion_strs)
    return tokenized_completions


if __name__ == "__main__":
    # Load the data
    train_dataset = load_dataset("corrigible-neutral-HHH.jsonl")

    # Load model
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "EleutherAI/pythia-70m"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_func(example):
        return tokenizer(example)

    # Tokenize the data
    train_dataset_tokenized = train_dataset.map(preprocess_func)

    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    wandb.finish()
