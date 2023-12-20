from repepo.core import Dataset
from repepo.core import Pipeline

from repepo.algorithms.base import Algorithm
from overrides import override

from dataclasses import dataclass, asdict
from dataclasses import field
from typing import List, Optional

import torch
import pprint
from torch.utils.data import DataLoader
from torch.optim import SGD
from transformers.optimization import Adafactor, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from repepo.core.types import Completion

from repepo.data import make_dataset, DatasetSpec
from repepo.data import utils
from repepo.data.dataset import sft
from repepo.utils.metrics import Metrics, AverageMeter
from repepo.variables import Environ
from repepo.variables import Model
from repepo.utils import logging
from repepo.core.benchmark import Benchmark

from accelerate import Accelerator


@dataclass
class SupervisedFineTuningConfig:
    # Training config
    batch_size: int = 256  # Training batch size
    shuffle: bool = True  # Whether to shuffle the dataset
    num_train_epochs: int = 10
    learning_rate: float = 5e-5

    # Experiment config
    device: str = "cuda"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: str = 'bf16'


def inspect_batch(batch: dict[str, torch.Tensor], tokenizer):
    
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]
    loss_active_idx = (labels != -100).nonzero()
    loss_inactive_idx = (labels == -100).nonzero()
    prompt_str = tokenizer.decode(input_ids[loss_inactive_idx].squeeze(-1), skip_special_tokens = True)
    prediction_str = tokenizer.decode(input_ids[loss_active_idx].squeeze(-1), skip_special_tokens = True)
    return {
        'prompt_str': prompt_str,
        'prediction_str': prediction_str
    }

    
class SupervisedFineTuning(Algorithm):
    @override
    def __init__(self, config: SupervisedFineTuningConfig):
        self.config = config

    def run(
        self,
        pipeline: Pipeline,
        dataset: Dataset,
        logger: Optional[logging.Logger] = None,
        eval_benchmark: Optional[Benchmark] = None,
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

        data_collator = sft.DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        # Make train dataloader
        completions: List[Completion] = pipeline.formatter.apply_list(dataset)
        _ds = sft.SupervisedDataset(completions, tokenizer=tokenizer)
        train_dataloader = DataLoader(
            _ds,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            collate_fn=data_collator,
        )

        # Make val dataloader
        eval_enabled = False
        if eval_benchmark is not None:
            eval_enabled = True
            val_dataset = eval_benchmark.test_dataset
            completions: List[Completion] = pipeline.formatter.apply_list(val_dataset)
            _ds = sft.SupervisedDataset(completions, tokenizer = tokenizer)
            val_dataloader = DataLoader(
                _ds,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                collate_fn=data_collator,
            )

        # Set up optimizer and scheduler
        num_training_steps = self.config.num_train_epochs * len(train_dataloader)
        optimizer = SGD(model.parameters(), lr=self.config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        accelerator = Accelerator(
            mixed_precision = self.config.mixed_precision,
            gradient_accumulation_steps= self.config.gradient_accumulation_steps
        )
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        if eval_enabled:
            val_dataloader = accelerator.prepare_data_loader(val_dataloader)

        # Training loop
        global_step = 0
        model.train()
        for epoch in range(int(self.config.num_train_epochs)):
            # epoch_iterator = tqdm(train_dataloader, desc="Training")
            epoch_iterator = train_dataloader   
            for step, batch in enumerate(epoch_iterator):
                with accelerator.accumulate(model):
                    global_step += self.config.batch_size
                    outputs = model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = outputs.loss
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    # Gradient clipping
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    if logger is not None:
                        logger.log({"train/loss": loss.item()}, step=global_step)

                        # Log the gradients for debugging
                        grads = [
                                param.grad.detach().flatten()
                                for param in model.parameters()
                                if param.grad is not None
                            ]
                        norm = torch.cat(grads).norm()
                        logger.log({"train/grad_norm": norm.item()}, step = global_step)

                    del outputs
                    del batch

            # TODO: Evaluation callback?
            if eval_enabled:
                model.eval()
                with torch.no_grad():
                    val_iterator = val_dataloader
                    for step, batch in enumerate(val_iterator):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            labels=batch["labels"],
                            attention_mask=batch["attention_mask"],
                        )
                        loss = outputs.loss
                        if logger is not None:
                            logger.log({"eval/loss": loss.item()}, step=global_step)

                        del outputs
                        del batch

                model.train()
        return pipeline


if __name__ == "__main__":
    import pyrallis
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from repepo.utils.logging import WandbConfig, WandbLogger
    from repepo.core.evaluate import BleuEvaluator, Rouge1Evaluator

    @dataclass
    class TrainSFTConfig:
        sft: SupervisedFineTuningConfig = SupervisedFineTuningConfig()
        dataset: DatasetSpec = DatasetSpec(name="truthfulqa", split = ":1%")
        val_dataset: DatasetSpec = DatasetSpec(name = "truthfulqa", split = "80:100%")
        wandb: WandbConfig = WandbConfig()

        model_name_or_path: str = "EleutherAI/pythia-1.4b"
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
        torch_dtype=torch.bfloat16,
        token=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        model_max_length=config.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    pipeline = Pipeline(model, tokenizer)
    eval_benchmark = Benchmark(
        'truthfulqa',
        make_dataset(config.dataset),
        make_dataset(config.val_dataset),
        []
    )

    with WandbLogger(config.wandb, extra_config=asdict(config)) as logger:
        algorithm = SupervisedFineTuning(config.sft)
        algorithm.run(pipeline, eval_benchmark.train_dataset, logger=logger, eval_benchmark=eval_benchmark)

        from repepo.core.benchmark import GenerationConfig
        eval_result = eval_benchmark.evaluate(
            pipeline, 
            generation_config = GenerationConfig(
                max_length = config.model_max_length,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
            )
        )

        

        # Log table of results
        # TODO(dtch1997): Additionally log per-sample metrics
        # TODO(dtch1997): refactor wandb dependency into logger
        import pandas as pd 
        import wandb 
        rows = []
        for prediction in eval_result.predictions:
            rows.append(dict(
                input = prediction.completion.prompt, 
                predicted_output = prediction.output,
                reference_output = prediction.completion.response
            ))
        eval_result_df = pd.DataFrame.from_records(rows)
        logger.log({"final_predictions": wandb.Table(dataframe = eval_result_df)})

        # Log metrics
        metrics = eval_result.metrics 
        metrics = {f"final_metrics/{k}": v for k, v in metrics.items()}
        logger.log(metrics)