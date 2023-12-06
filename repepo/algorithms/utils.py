from repepo.core import BaseDataset, BasePipeline

from typing import List
from repepo.data.dataset import sft
from torch.utils.data import DataLoader


def prepare_dataloader(
    pipeline: BasePipeline, dataset: BaseDataset, batch_size: int, shuffle: bool = False
) -> DataLoader:
    examples: List[format.Example] = dataset.examples
    # Format the dataset
    completions: List[format.Completion] = pipeline.formatter.apply(examples)
    completions: List[format.Completion] = pipeline.prompter.apply(completions)

    supervised_dataset = sft.SupervisedDataset(
        completions, tokenizer=pipeline.tokenizer
    )
    data_collator = sft.DataCollatorForSupervisedDataset(tokenizer=pipeline.tokenizer)

    return DataLoader(
        supervised_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        shuffle=shuffle,
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
