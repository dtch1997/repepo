from repepo.core import Dataset
from repepo.core import Pipeline

from repepo.algorithms.base import Algorithm
from overrides import override

from dataclasses import dataclass
from dataclasses import field
from typing import List

import torch
import pprint
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from repepo.core.types import Completion

from repepo.data import make_dataset, DatasetSpec
from repepo.data import utils
from repepo.data.dataset import sft
from repepo.utils.metrics import Metrics
from repepo.variables import Environ
from repepo.variables import Model
from repepo.algorithms.utils import AverageMeter


@dataclass
class SupervisedFineTuningConfig:
    # Training config
    batch_size: int = 256  # Training batch size
    shuffle: bool = True  # Whether to shuffle the dataset
    num_train_epochs: int = 10
    learning_rate: float = 5e-5

    # Experiment config
    device: str = "cuda"


@dataclass
class WandbConfig:
    project: str = field(default=Environ.WandbProject)
    entity: str = field(default=Environ.WandbEntity)
    name: str = field(default="sft-simple")
    track: bool = field(default=False)


class WandbLogger:
    def __init__(self, config: WandbConfig):
        self.config = config
        if self.config.track:
            import wandb

            self.wandb = wandb

    def __enter__(self):
        if self.config.track:
            self.wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.name,
            )
        return self

    def log(self, *args, **kwargs):
        if self.config.track:
            self.wandb.log(*args, **kwargs)
        # Else no-op

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config.track:
            self.wandb.finish()


class SupervisedFineTuning(Algorithm):
    @override
    def __init__(self, config: SupervisedFineTuningConfig):
        self.config = config

    def run(
        self, pipeline: Pipeline, dataset: Dataset, logger: WandbLogger
    ) -> Pipeline:
        """Modifies the base model weights"""

        # Load model and tokenizer
        model = pipeline.model
        tokenizer = pipeline.tokenizer

        # Add new tokens to tokenizer
        # This is because many tokenizers don't have a padding token
        special_tokens_dict = utils.get_pad_token(tokenizer)
        special_tokens_dict.update(utils.get_special_tokens(tokenizer))
        utils.smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        # Make train dataloader
        completions: List[Completion] = pipeline.formatter.apply_list(dataset)
        _ds = sft.SupervisedDataset(completions, tokenizer=tokenizer)
        data_collator = sft.DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        train_dataloader = DataLoader(
            _ds,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            collate_fn=data_collator,
        )

        # Set device
        device = self.config.device if torch.cuda.is_available() else "cpu"
        model.to(device)  # type: ignore

        # Set up optimizer and scheduler
        num_training_steps = self.config.num_train_epochs * len(train_dataloader)
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # Set up metrics, meters
        metric_fns = Metrics()
        meter = AverageMeter()

        # Training loop
        global_step = 0
        model.train()
        for epoch in range(int(self.config.num_train_epochs)):
            # epoch_iterator = tqdm(train_dataloader, desc="Training")
            epoch_iterator = train_dataloader
            for step, batch in enumerate(epoch_iterator):
                global_step += self.config.batch_size
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                print(f"epoch : {epoch} | step: {step} | loss: {loss}")
                logger.log({"train/loss": loss}, step=global_step)

            # TODO: Evaluation callback?

        return pipeline


if __name__ == "__main__":
    import pyrallis
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    @dataclass
    class TrainSFTConfig:
        sft: SupervisedFineTuningConfig = SupervisedFineTuningConfig()
        dataset: DatasetSpec = DatasetSpec(name="truthfulqa")
        wandb: WandbConfig = WandbConfig()

        model_name_or_path: str = Model.Pythia70m
        model_max_length: int = field(
            default=512,
            metadata={
                "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            },
        )
        cache_dir: str = Environ.HuggingfaceCacheDir
        output_dir: str = Environ.OutputDir

    config = pyrallis.parse(TrainSFTConfig)
    pprint.pprint(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
    )

    # TODO: figure out typing for this
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        model_max_length=config.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    pipeline = Pipeline(model, tokenizer)
    dataset = make_dataset(config.dataset)

    with WandbLogger(config.wandb) as logger:
        algorithm = SupervisedFineTuning(config.sft)
        algorithm.run(pipeline, dataset, logger=logger)
