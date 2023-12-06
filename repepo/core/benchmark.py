# pyright: strict, reportMissingTypeStubs=false

from dataclasses import dataclass
from typing import Optional, Sequence

from transformers import GenerationConfig

from repepo.algorithms.base import BaseAlgorithm
from repepo.core.evaluate import EvalPrediction, EvalResult, Evaluator
from repepo.core.format import AbstractFormatter, InputOutputFormatter
from repepo.core.pipeline import Pipeline

from repepo.core.types import Dataset, Model, Tokenizer


@dataclass
class Benchmark:
    name: str
    train_dataset: Dataset
    test_dataset: Dataset
    evaluators: list[Evaluator]


def train_and_evaluate_benchmark(
    model: Model,
    tokenizer: Tokenizer,
    algorithms: Sequence[BaseAlgorithm],
    benchmark: Benchmark,
    formatter: Optional[AbstractFormatter] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> EvalResult:
    # set up pipeline
    pipeline = Pipeline(model, tokenizer, formatter=formatter or InputOutputFormatter())

    # train pipeline
    for algorithm in algorithms:
        pipeline = algorithm.run(pipeline, benchmark.train_dataset)

    # evaluate
    predictions: list[EvalPrediction] = []
    # TODO: support batching
    for example in benchmark.test_dataset:
        output = pipeline.generate(example, generation_config=generation_config)
        predictions.append(EvalPrediction(example, output))
    metrics: dict[str, float] = {}
    for evaluator in benchmark.evaluators:
        metrics.update(evaluator(predictions))
    return EvalResult(predictions, metrics)
