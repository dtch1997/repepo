import evaluate

from typing import Callable, Dict, List
from .base import Pipeline
from repepo.core.types import Dataset
from .utils import AverageMeter

Callback = Callable[[Pipeline], Dict[str, float]]


class Metrics:
    def __init__(self):
        # TODO: make configurable
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")

    def compute_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        bleu_results = self.bleu.compute(predictions=predictions, references=references)
        rouge_results = self.rouge.compute(
            predictions=predictions, references=references
        )
        assert bleu_results is not None
        assert rouge_results is not None
        return {
            "bleu": bleu_results["bleu"],
            "rouge1": rouge_results["rouge1"],
        }


class EvalCallback:
    def __init__(self, val_datasets: Dict[str, Dataset]):
        self.metric_fns = Metrics()
        self.meter = AverageMeter()
        # TODO: eval dataloader

    def __call__(self, pipeline: Pipeline) -> Dict[str, float]:
        self.meter.reset()
        model = pipeline.model
        tokenizer = pipeline.tokenizer
        log_dict = {}

        # TODO: implement

        return log_dict
