from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored
from torch.utils.data import DataLoader
from repepo.core.types import Tokenizer

# Load pre-trained model and tokenizer from Huggingface
from repepo.data import get_dataset
from repepo.data import utils
from repepo.data.dataset import format
from repepo.data.dataset import sft
from repepo.utils.metrics import Metrics
from repepo.variables import Environ
from repepo.variables import Model

# keep pyright happy
assert Environ.WandbEntity is not None


@dataclass
class WandbArguments:
    entity: str = field(default=Environ.WandbEntity)
    project: str = field(default=Environ.WandbProject)
    name: str = field(default="sft-simple")
    track: bool = field(default=False)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=Model.Pythia70m)


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="truthfulqa",
        metadata={
            "help": "Name of training dataset. See repepo.data.list_datasets() for details"
        },
    )
    batch_size: int = field(default=32)
    train_fraction: float = field(default=0.0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=Environ.HuggingfaceCacheDir)
    output_dir: Optional[str] = field(default=Environ.OutputDir)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def make_supervised_data_module(tokenizer: Tokenizer, data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    # Get the dataset by name
    examples: List[format.Example] = get_dataset(data_args.dataset_name)
    # Format the dataset
    completions: List[format.Completion] = format.QAFormatter().apply(examples)
    completions = format.FewShotPrompter(n_few_shot_examples=10).apply(completions)

    # Initialize dataset
    eval_dataset = sft.SupervisedDataset(completions, tokenizer=tokenizer)
    data_collator = sft.DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Make data loader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    return dict(
        eval_dataloader=eval_dataloader,
    )


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
            self.metrics[name] = {"val": 0, "sum": 0, "count": 0, "avg": 0}

        metric = self.metrics[name]
        metric["val"] = value
        metric["sum"] += value * n
        metric["count"] += n
        metric["avg"] = metric["sum"] / metric["count"]

    def get_avg(self, name):
        """
        Get the running average of a named metric.

        Parameters:
        - name: The name of the metric.
        """
        return self.metrics[name]["avg"] if name in self.metrics else None

    def reset(self, name=None):
        """
        Resets statistics of a named metric or all metrics if name is None.

        Parameters:
        - name: The name of the metric.
        """
        if name:
            self.metrics[name] = {"val": 0, "sum": 0, "count": 0, "avg": 0}
        else:
            for metric in self.metrics.values():
                metric["val"] = 0
                metric["sum"] = 0
                metric["count"] = 0
                metric["avg"] = 0


def eval_model():
    """Simple Pytorch-only eval loop"""

    # Parse CLI args
    # TODO: figure out typing for this
    parser_args: Any = (
        ModelArguments,
        DataArguments,
        TrainingArguments,
        WandbArguments,
    )
    parser = transformers.HfArgumentParser(parser_args)
    (
        model_args,
        data_args,
        training_args,
        wandb_args,
    ) = parser.parse_args_into_dataclasses()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
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
    data_module = make_supervised_data_module(tokenizer, data_args)
    eval_dataloader = data_module["eval_dataloader"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up metrics, meters
    metric_fns = Metrics()
    meter = AverageMeter()

    wandb = None
    if wandb_args.track:
        import wandb

        wandb.init(
            project=wandb_args.project, entity=wandb_args.entity, name=wandb_args.name
        )

    global_step = 0
    with torch.no_grad():
        model.eval()
        meter.reset()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Calculate val loss
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            loss = outputs.loss
            if wandb_args.track and wandb is not None:
                wandb.log({"eval/loss": loss}, step=global_step)

            prediction_ids = model.generate(
                batch["prompt_ids"],
                max_length=training_args.model_max_length,
                temperature=0.3,  # TODO: make configurable
                do_sample=True,
                # Do this to suppress warning:
                # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id/71397707#71397707
                pad_token_id=tokenizer.eos_token_id,
            )
            prompts = [
                tokenizer.decode(prompt, skip_special_tokens=True)
                for prompt in batch["prompt_ids"]
            ]
            predictions = [
                tokenizer.decode(pred, skip_special_tokens=True)
                for pred in prediction_ids
            ]
            # Remove the prompt text from predictions
            predictions = [
                pred[len(prompt) :] for prompt, pred in zip(prompts, predictions)
            ]
            references = [
                tokenizer.decode(ref, skip_special_tokens=True)
                for ref in batch["reference_ids"]
            ]
            # Visualize predictions
            print("Sample prompt: ", colored(prompts[0], "yellow"))
            print()
            print("Predicted answer: ", colored(predictions[0], "light_blue"))
            print()
            print("Reference answer: ", colored(references[0], "green"))
            print()
            metrics = metric_fns.compute_metrics(predictions, references)
            for name, metric in metrics.items():
                meter.update(name, metric)

        for name, metric in meter.metrics.items():
            val = metric["avg"]
            print(f"Average {name}: {val}")
            if wandb_args.track and wandb is not None:
                wandb.log({f"eval/{name}": val}, step=global_step)

    # Save the fine-tuned model
    # model.save_pretrained("./fine_tuned_model")
    # tokenizer.save_pretrained("./fine_tuned_model")


if __name__ == "__main__":
    eval_model()
