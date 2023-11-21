from typing import Dict, List

import evaluate
import numpy as np


class Metrics:
    def __init__(self):
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
