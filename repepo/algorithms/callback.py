import evaluate
import numpy as np

from typing import Callable, Dict, List
from .base import BasePipeline, BaseDataset
from .utils import AverageMeter

Callback = Callable[[BasePipeline], Dict[str, float]]


class Metrics:
    def __init__(self):
        # TODO: make configurable
        self.bleu = evaluate.load("bleu")
        self.bleurt = evaluate.load("bleurt", module_type="metric")
        self.rouge = evaluate.load("rouge")

    def compute_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        bleu_results = self.bleu.compute(predictions=predictions, references=references)
        bleurt_results = self.bleurt.compute(
            predictions=predictions, references=references
        )
        rouge_results = self.rouge.compute(
            predictions=predictions, references=references
        )
        return {
            "bleu": bleu_results["bleu"],
            "bleurt": np.mean(bleurt_results["scores"]),
            "rouge1": rouge_results["rouge1"],
        }


class EvalCallback:
    def __init__(self, val_datasets: Dict[str, BaseDataset]):
        self.metric_fns = Metrics()
        self.meter = AverageMeter()
        self.temperature = 0.3  # TODO: make configurable
        # TODO: make datasets configurable?

    def __call__(self, pipeline: BasePipeline) -> Dict[str, float]:
        self.meter.reset()
        model = pipeline.model
        tokenizer = pipeline.tokenizer
        log_dict = {}

        for batch in self.eval_dataloader:
            batch = {k: v.to(pipeline.device) for k, v in batch.items()}

            # Calculate val loss
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            log_dict["loss"] = outputs.loss

            # Calculate other metrics
            prediction_ids = model.generate(
                batch["prompt_ids"],
                max_length=pipeline.config.model_max_length,
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
            metrics = self.metric_fns.compute_metrics(predictions, references)
            for name, metric in metrics.items():
                self.meter.update(name, metric)
