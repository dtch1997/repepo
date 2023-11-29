# pyright: strict, reportMissingTypeStubs=false

from dataclasses import dataclass
from typing import Generic, Optional, Sequence

from transformers import GenerationConfig

from repepo.algorithms.base import BaseAlgorithm
from repepo.core.evaluate import EvalResT, Evaluator, OverallEvalResults
from repepo.core.format import AbstractFormatter, InputOutputFormatter
from repepo.core.pipeline import Pipeline

from repepo.core.types import Dataset, Model, Tokenizer


@dataclass
class Benchmark(Generic[EvalResT]):
    name: str
    train_dataset: Dataset
    test_dataset: Dataset
    evaluator: Evaluator[EvalResT]


def train_and_evaluate_benchmark(
    model: Model,
    tokenizer: Tokenizer,
    algorithms: Sequence[BaseAlgorithm],
    benchmark: Benchmark[EvalResT],
    formatter: Optional[AbstractFormatter] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> OverallEvalResults[EvalResT]:
    # set up pipeline
    pipeline = Pipeline(model, tokenizer, formatter=formatter or InputOutputFormatter())

    # train pipeline
    for algorithm in algorithms:
        pipeline = algorithm.run(pipeline, benchmark.train_dataset)

    # evaluate
    eval_results: list[EvalResT] = []
    # TODO: support batching
    for example in benchmark.test_dataset:
        output = pipeline.generate(example, generation_config=generation_config)
        eval_results.append(benchmark.evaluator(example, output))
    return OverallEvalResults(eval_results)
