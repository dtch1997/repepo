from repepo.core import benchmark
from repepo.data import make_dataset, DatasetSpec
from typing import Sequence

from repepo.core.types import Dataset, Model, Tokenizer


def make_tqa_mc1_random_benchmark(seed: int = 0) -> benchmark.Benchmark:
    train_dataset: Dataset = make_dataset(
        DatasetSpec(
            name="truthfulqa",
            split=":80%",
            seed=seed,
        )
    )

    test_dataset: Dataset = make_dataset(
        DatasetSpec(
            name="truthfulqa",
            split="80%:",
            seed=seed,
        )
    )

    return benchmark.Benchmark(
        name="tqa_mc1",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        evaluators=[],  # Evaluator defined elsewhere
    )


def evaluate_mc1(
    model: Model,
    tokenizer: Tokenizer,
    algorithms: Sequence[Algorithm],
    benchmark: Benchmark,
    formatter: Optional[Formatter] = None,
    generation_config: Optional[GenerationConfig] = None,
):
    """Evaluates a model on a MC1-style benchmark."""
    pipeline = Pipeline(model, tokenizer, formatter=formatter or InputOutputFormatter())
    for algorithm in algorithms:
        pipeline = algorithm.run(pipeline, benchmark.train_dataset)

    for example in benchmark.test_dataset:
        # Get log-probabilities of correct answer
        pass
