import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Load pre-trained model and tokenizer from Huggingface
from transformers import AutoModelForCausalLM, AutoTokenizer


from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

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
    batch_size: int = field(default = 32)
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

def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor, 
    max_length: int =50, 
    temperature: float =0.7
) -> List[str]:
    """
    Generate a text completion given a prompt using a pre-trained causal language model.
    
    Parameters:
    - input_ids: torch.Tensor of shape (B, T), token IDs describing the prompt
    - model_name: str, identifier of the pre-trained model
    - max_length: int, maximum length of the generated text
    - temperature: float, sampling temperature for diversity (higher values make output more random)
    
    Returns:
    - str, generated text completion
    """
  
    # Generate text completion
    completion = model.generate(
        input_ids,
        max_length=max_length, 
        temperature=temperature, 
        pad_token_id=tokenizer.pad_token_id
    )

    # Step 1: Get the total length of the input tokens
    input_length = input_ids.shape[-1]
    # Step 2: Slice the completion tensor to keep only the generated tokens, omitting the input tokens
    generated_tokens = completion[:, input_length:]
    # Step 3: If we have a batch, select the generated tokens of the first sequence
    generated_tokens_first_sequence = generated_tokens[0]
    # Step 4: Decode the token IDs of the generated tokens to a readable string, skipping special tokens
    completion_text = tokenizer.decode(generated_tokens_first_sequence, skip_special_tokens=True)
        
    return completion_text

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

    # Add new tokens to tokenizer
    special_tokens_dict = utils.get_pad_token(tokenizer)
    special_tokens_dict.update(utils.get_special_tokens(tokenizer))
    utils.smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
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

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 3)

    # Training loop
    model.train()
    for epoch in range(int(training_args.num_train_epochs)):
        # epoch_iterator = tqdm(train_dataloader, desc="Training")
        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):

            batch = {k : v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids = batch['input_ids'],
                labels = batch['labels'],
                attention_mask = batch['attention_mask']
            )
            loss = outputs.loss
            loss.backward()
            print(f"epoch : {epoch} | step: {step} | loss: {loss}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    # Save the fine-tuned model
    # model.save_pretrained("./fine_tuned_model")
    # tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    train_model()