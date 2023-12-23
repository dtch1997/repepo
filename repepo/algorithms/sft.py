from repepo.core import Dataset
from repepo.core import Pipeline

from repepo.algorithms.base import Algorithm
from overrides import override

from dataclasses import dataclass
from dataclasses import field
from typing import List, Optional

import os
import pprint
import pyrallis
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from repepo.utils.logging import WandbConfig, WandbLogger
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments as HfTrainingArguments

from repepo.core.types import Completion

from repepo.data import make_dataset, DatasetSpec
from repepo.data.dataset import sft
from repepo.variables import Environ
from repepo.variables import Model

from typing import Any


@dataclass
class SupervisedFineTuningConfig(HfTrainingArguments):
    output_dir: Optional[str] = field(default=Environ.OutputDir)
    optim: str = field(default="adamw_torch")
    report_to: List[str] = field(default_factory=lambda: ["wandb"])


class SupervisedFineTuning(Algorithm):
    @override
    def __init__(self, config: SupervisedFineTuningConfig):
        self.config = config

    def run(
        self,
        pipeline: Pipeline,
        dataset: Dataset,
    ) -> dict[str, Any]:
        """Modifies the base model weights

        Returns a dict of metrics"""

        # Set padding token if necessary
        tokenizer = pipeline.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Make dataset
        completions: List[Completion] = pipeline.formatter.apply_list(dataset)
        train_dataset = sft.SupervisedDataset(completions, tokenizer=tokenizer)
        data_collator = sft.DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        trainer = Trainer(
            model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            args=self.config,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        trainer.train()

        train_state_log_history: List[dict[str, float]] = trainer.state.log_history
        return {
            "train_state_log_history": train_state_log_history,
        }


@dataclass
class TrainSFTConfig:
    sft: SupervisedFineTuningConfig = field(
        default_factory=lambda: SupervisedFineTuningConfig()
    )
    dataset: DatasetSpec = DatasetSpec(name="stereoset", split=":80%", seed=0)
    wandb: WandbConfig = WandbConfig()

    model_name_or_path: str = Model.Pythia70m
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    cache_dir: str = Environ.HuggingfaceCacheDir


def setup_wandb_env_variables(config: TrainSFTConfig):
    # if not config.wandb.track:
    #     os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_WATCH"] = "all"


def train_sft(config: TrainSFTConfig):
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

    setup_wandb_env_variables(config)
    if not config.wandb.track:
        config.sft.report_to = []
    else:
        config.sft.report_to = ["wandb"]
    with WandbLogger(config.wandb):
        algorithm = SupervisedFineTuning(config.sft)
        metrics = algorithm.run(pipeline, dataset)
    return metrics


if __name__ == "__main__":
    config = pyrallis.parse(TrainSFTConfig)
    pprint.pprint(config)
    metrics = train_sft(config)
    breakpoint()
