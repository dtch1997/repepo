# pyright: strict, reportMissingTypeStubs=false

from dataclasses import dataclass, replace
from typing import Optional, Sequence

from transformers.generation import GenerationConfig

from repepo.algorithms.base import Algorithm
from repepo.core.evaluate import EvalPrediction, EvalResult, Evaluator
from repepo.core.format import Formatter, InputOutputFormatter
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
    algorithms: Sequence[Algorithm],
    benchmark: Benchmark,
    formatter: Optional[Formatter] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> EvalResult:
    pipeline = Pipeline(model, tokenizer, formatter=formatter or InputOutputFormatter())

    # train pipeline
    for algorithm in algorithms:
        # Re-initialize pipeline, which gets destructively modified
        output = algorithm.run(pipeline, benchmark.train_dataset)

    # evaluate
    predictions: list[EvalPrediction] = []
    requires_generation = any([e.requires_generation for e in benchmark.evaluators])
    requires_probs = any([e.requires_probs for e in benchmark.evaluators])
    # TODO: support batching
    for example in benchmark.test_dataset:
        generated_output = None
        correct_output_probs = None
        incorrect_outputs_probs = None
        if requires_generation:
            generated_output = pipeline.generate(
                example, generation_config=generation_config
            )
        if requires_probs:
            correct_output_probs = pipeline.calculate_output_logprobs(example)
            if example.incorrect_outputs is not None:
                incorrect_outputs_probs = [
                    pipeline.calculate_output_logprobs(
                        replace(example, output=incorrect_output)
                    )
                    for incorrect_output in example.incorrect_outputs
                ]
        predictions.append(
            EvalPrediction(
                example=example,
                generated_output=generated_output,
                correct_output_probs=correct_output_probs,
                incorrect_outputs_probs=incorrect_outputs_probs,
            )
        )
    metrics: dict[str, float] = {}
    for evaluator in benchmark.evaluators:
        metrics.update(evaluator(predictions))
    return EvalResult(predictions, metrics)
