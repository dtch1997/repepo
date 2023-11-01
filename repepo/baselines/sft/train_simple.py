import torch
import evaluate
import numpy as np
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
class WandbArguments:
    entity: str = field(default = Environ.WandbEntity)
    project: str = field(default = Environ.WandbProject)
    name: str = field(default = "sft")
    track: bool = field(default = False)

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

class Metrics:
    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.bleurt = evaluate.load("bleurt", module_type="metric")
        self.rouge = evaluate.load("rouge")

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        bleu_results = self.bleu.compute(predictions=predictions, references=references)
        bleurt_results = self.bleurt.compute(predictions=predictions, references=references)
        rouge_results = self.rouge.compute(predictions=predictions, references=references)
        return {
            'bleu': bleu_results['bleu'],
            'bleurt': np.mean(bleurt_results['scores']),
            'rouge1': rouge_results['rouge1']
        }
    
class AverageMeter:
    def __init__(self):
        self.metrics = {}
        
    def update(self, name, value, n=1):
        """
        Update a named metric with a new value.
        
        Parameters:
        - name: The name of the metric.
        - value: The new value to incorporate.
        - n: The weight/count of the value. Default is 1.
        """
        if name not in self.metrics:
            self.metrics[name] = {'val': 0, 'sum': 0, 'count': 0, 'avg': 0}
        
        metric = self.metrics[name]
        metric['val'] = value
        metric['sum'] += value * n
        metric['count'] += n
        metric['avg'] = metric['sum'] / metric['count']
        
    def get_avg(self, name):
        """
        Get the running average of a named metric.
        
        Parameters:
        - name: The name of the metric.
        """
        return self.metrics[name]['avg'] if name in self.metrics else None
    
    def reset(self, name=None):
        """
        Resets statistics of a named metric or all metrics if name is None.
        
        Parameters:
        - name: The name of the metric.
        """
        if name:
            self.metrics[name] = {'val': 0, 'sum': 0, 'count': 0, 'avg': 0}
        else:
            for metric in self.metrics.values():
                metric['val'] = 0
                metric['sum'] = 0
                metric['count'] = 0
                metric['avg'] = 0

def train_model():
    """ Simple PyTorch-only training loop """
    
    # Parse CLI args
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, WandbArguments))
    model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

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
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 3)

    # Set up metrics, meters
    metric_fns = Metrics()
    meter = AverageMeter()

    if wandb_args.track:
        import wandb
        wandb.init(
            project=wandb_args.project,
            entity=wandb_args.entity,
            name=wandb_args.name
        )

    # Training loop
    global_step = 0
    model.train()
    for epoch in range(int(training_args.num_train_epochs)):
        # epoch_iterator = tqdm(train_dataloader, desc="Training")
        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):
            global_step += data_args.batch_size
            batch = {k : v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids = batch['input_ids'],
                labels = batch['labels'],
                attention_mask = batch['attention_mask']
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()            
            print(f"epoch : {epoch} | step: {step} | loss: {loss}")
            if wandb_args.track:
                wandb.log({'train/loss': loss}, step=global_step)

        # Evaluation step (after each epoch)
        with torch.no_grad():

            meter.reset()
            for batch in eval_dataloader:
                batch = {k : v.to(device) for k, v in batch.items()}
                prediction_ids = model.generate(
                    batch['prompt_ids'],
                    max_length = training_args.model_max_length,
                    temperature=0.3, # TODO: make configurable
                    do_sample=True,
                    # Do this to suppress warning: 
                    # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id/71397707#71397707
                    pad_token_id = tokenizer.eos_token_id, 
                )
                prompts = [tokenizer.decode(prompt, skip_special_tokens=True) for prompt in batch['prompt_ids']]
                predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in prediction_ids]
                # Remove the prompt text from predictions
                predictions = [pred[len(prompt):] for prompt, pred in zip(prompts, predictions)]
                references = [tokenizer.decode(ref, skip_special_tokens=True) for ref in batch['reference_ids']]
                # Visualize predictions
                print("Sample prompt: ", prompts[0])
                print()
                print("Predicted answer: ", predictions[0])
                print("Reference answer: ", references[0])
                print()
                metrics = metric_fns.compute_metrics(predictions, references)
                for name, metric in metrics.items():
                    meter.update(name, metric)
                
            for name, metric in meter.metrics.items():
                val = metric['avg']
                print(f"Average {name}: {val}")
                if wandb_args.track:
                    wandb.log({f'eval/{name}': val}, step=global_step)
                

    # Save the fine-tuned model
    # model.save_pretrained("./fine_tuned_model")
    # tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    train_model()