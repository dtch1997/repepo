# pyright: strict, reportMissingTypeStubs=false

from dataclasses import dataclass
from typing import Optional, Sequence

from transformers.generation import GenerationConfig # keep pyright happy

from repepo.algorithms.base import Algorithm
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

    def evaluate(self, pipeline: Pipeline, generation_config: Optional[GenerationConfig] = None,) -> EvalResult:
        predictions: list[EvalPrediction] = []
        # TODO: support batching
        for example in self.test_dataset:
            completion = pipeline.formatter.apply(example)
            completion = pipeline.prompter.apply(completion)
            output = pipeline.generate(example, generation_config=generation_config)
            predictions.append(EvalPrediction(example, completion, output))
        metrics: dict[str, float] = {}
        for evaluator in self.evaluators:
            metrics.update(evaluator(predictions))
        return EvalResult(predictions, metrics)

def train_and_evaluate_benchmark(
    model: Model,
    tokenizer: Tokenizer,
    algorithms: Sequence[Algorithm],
    benchmark: Benchmark,
    formatter: Optional[AbstractFormatter] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> EvalResult:
    # set up pipeline
    pipeline = Pipeline(model, tokenizer, formatter=formatter or InputOutputFormatter())

    # train pipeline
    for algorithm in algorithms:
        pipeline = algorithm.run(pipeline, benchmark.train_dataset)

    return benchmark.evaluate(pipeline, generation_config=generation_config)
