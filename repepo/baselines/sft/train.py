#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Optional

from transformers.trainer import Trainer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.hf_argparser import HfArgumentParser
from repepo.core.types import Tokenizer

from repepo.data import get_dataset
from repepo.data.dataset import sft
from repepo.variables import Environ
from repepo.variables import Model
from repepo.core.types import Completion, Example
from repepo.core.format import InstructionFormatter


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=Model.Pythia70m)


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="stereoset",
        metadata={
            "help": "Name of training dataset. See repepo.data.list_datasets() for details"
        },
    )


@dataclass
class TrainingArguments(HfTrainingArguments):
    cache_dir: Optional[str] = field(default=Environ.HuggingfaceCacheDir)
    output_dir: Optional[str] = field(default=Environ.OutputDir)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def make_supervised_data_module(
    tokenizer: Tokenizer, data_args: DataArguments
) -> tuple[sft.SupervisedDataset, sft.DataCollatorForSupervisedDataset]:
    """Make dataset and collator for supervised fine-tuning."""
    examples: list[Example] = get_dataset(data_args.dataset_name)
    fmt = InstructionFormatter()
    completions: list[Completion] = [
        fmt.format_conversation([ex])[0] for ex in examples
    ]
    train_dataset = sft.SupervisedDataset(completions, tokenizer=tokenizer)
    data_collator = sft.DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return train_dataset, data_collator


def train():
    # import logging
    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # for logger in loggers:
    #     logger.setLevel(logging.INFO)
    # TODO: figure out typing for this
    # TODO: Are there off-by-one errors in SFT dataset?
    parser_args: Any = (ModelArguments, DataArguments, TrainingArguments)
    parser = HfArgumentParser(parser_args)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # TODO: figure out typing for this
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # special_tokens_dict = utils.get_pad_token(tokenizer)
    # special_tokens_dict.update(utils.get_special_tokens(tokenizer))
    # utils.smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model,
    # )

    # TODO: Is it principled to set the pad token to the eos token?
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, data_collator = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args
    )

    # Train
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
