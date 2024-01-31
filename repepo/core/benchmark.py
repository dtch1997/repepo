# pyright: strict, reportMissingTypeStubs=false

from dataclasses import dataclass
from typing import Optional, Sequence

from transformers.generation import GenerationConfig

from repepo.algorithms.base import Algorithm
from repepo.core.evaluate import (
    EvalHook,
    EvalResult,
    Evaluator,
    evaluate,
)
from repepo.core.format import Formatter, InputOutputFormatter
from repepo.core.pipeline import Pipeline

from repepo.core.types import Dataset, Model, Tokenizer


@dataclass
class Benchmark:
    name: str
    train_dataset: Dataset
    test_dataset: Dataset
    evaluators: list[Evaluator]


def train_benchmark(
    model: Model,
    tokenizer: Tokenizer,
    algorithms: Sequence[Algorithm],
    benchmark: Benchmark,
    formatter: Optional[Formatter] = None,
) -> Pipeline:
    pipeline = Pipeline(
        model,
        tokenizer,
        formatter=formatter or InputOutputFormatter(),
    )
    for algorithm in algorithms:
        # Re-initialize pipeline, which gets destructively modified
        # TODO: do something with outputs?
        _ = algorithm.run(pipeline, benchmark.train_dataset)
    return pipeline


def evaluate_benchmark(
    pipeline: Pipeline,
    benchmark: Benchmark,
    generation_config: Optional[GenerationConfig] = None,
    # these eval_hooks allow us to do custom stuff to the pipeline only during evaluation,
    # e.g. mess with the formatter to use CAA's special answer format
    eval_hooks: list[EvalHook] = [],
    show_progress: bool = True,
    tqdm_desc: str = "Evaluating",
) -> EvalResult:
    # evaluate
    return evaluate(
        pipeline,
        dataset=benchmark.test_dataset,
        evaluators=benchmark.evaluators,
        generation_config=generation_config,
        eval_hooks=eval_hooks,
        show_progress=show_progress,
        tqdm_desc=tqdm_desc,
    )


def train_and_evaluate_benchmark(
    model: Model,
    tokenizer: Tokenizer,
    algorithms: Sequence[Algorithm],
    benchmark: Benchmark,
    formatter: Optional[Formatter] = None,
    generation_config: Optional[GenerationConfig] = None,
    # these eval_hooks allow us to do custom stuff to the pipeline only during evaluation,
    # e.g. mess with the formatter to use CAA's special answer format
    eval_hooks: list[EvalHook] = [],
    show_progress: bool = True,
) -> EvalResult:
    # train
    pipeline = train_benchmark(
        model,
        tokenizer,
        algorithms,
        benchmark,
        formatter=formatter,
    )

    # evaluate
    return evaluate_benchmark(
        pipeline,
        benchmark,
        generation_config=generation_config,
        eval_hooks=eval_hooks,
        show_progress=show_progress,
    )
