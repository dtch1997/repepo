import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Load pre-trained model and tokenizer from Huggingface
from transformers import AutoModelForSequenceClassification, AutoTokenizer


from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import transformers
from transformers import Trainer

from repepo.data import utils, get_dataset
from repepo.data.dataset import sft
from repepo.variables import Environ, Model

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=Model.Pythia70m)

@dataclass
class DataArguments:
    dataset_name: str = field(default="truthfulqa", metadata={"help": "Name of training dataset. See repepo.data.list_datasets() for details"})
    batch_size: int = field(default = 256)
    train_fraction: float = field(default = 0.8)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=Environ.HuggingfaceCacheDir)
    output_dir: Optional[str] = field(default=Environ.OutputDir)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    # Get the dataset by name
    list_data_dict = get_dataset(data_args.dataset_name)

    # Split the dataset
    num_examples = len(list_data_dict)
    train_size = int(num_examples * data_args.train_fraction)
    val_size = num_examples - train_size
    train_data, val_data = list_data_dict[:train_size], list_data_dict[train_size:]

    # Initialize datasets
    train_dataset = sft.SupervisedDataset(train_data, tokenizer=tokenizer)
    eval_dataset = sft.SupervisedDataset(val_data, tokenizer=tokenizer)
    data_collator = sft.DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Make data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size= data_args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size= data_args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    return dict(
        train_dataloader=train_dataloader, 
        eval_dataloader=eval_dataloader,
    )

def train_model():
    """ Simple PyTorch-only training loop """
    
    # Parse CLI args
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # Load dataset
    data_module = make_supervised_data_module(
        tokenizer, data_args
    )
    train_dataloader = data_module['train_dataloader']
    eval_dataloader = data_module['eval_dataloader']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # # Set up optimizer and scheduler
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 3)

    # # Training loop
    # model.train()
    # for epoch in range(3):
    #     epoch_iterator = tqdm(train_dataloader, desc="Training")
    #     for step, batch in enumerate(epoch_iterator):
    #         batch = tuple(t.to(device) for t in batch)
    #         inputs = {
    #             "input_ids": batch[0],
    #             "attention_mask": batch[1],
    #             "labels": batch[2],
    #         }
            
    #         outputs = model(**inputs)
    #         loss = outputs.loss
    #         loss.backward()
            
    #         # Gradient clipping
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
    #         optimizer.step()
    #         scheduler.step()
    #         model.zero_grad()

    # # Save the fine-tuned model
    # model.save_pretrained("./fine_tuned_model")
    # tokenizer.save_pretrained("./fine_tuned_model")
