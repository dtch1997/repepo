import pytest
from transformers import GenerationConfig

from repepo.algorithms.icl import InContextLearning
from repepo.core.benchmark import Benchmark, train_and_evaluate_benchmark
from repepo.core.evaluate import AccuracyEvaluator
from repepo.core.types import Dataset, Example, Tokenizer, Model


def test_evaluate_benchmark(larger_model: Model, larger_tokenizer: Tokenizer) -> None:
    train_dataset: Dataset = [
        Example("", "Paris is located in the country of", "France"),
        Example("", "Shanghai is located in the country of", "China"),
        Example("", "Tokyo is located in the country of", "Japan"),
        Example("", "London is located in the country of", "England"),
    ]
    test_dataset = [
        Example("", "Kyoto is located in the country of", "Japan"),
        Example("", "Beijing is located in the country of", "China"),
        Example("", "FakePlace is located in the country of", "WrongAnswer"),
    ]
    benchmark = Benchmark(
        name="test benchmark",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        evaluators=[AccuracyEvaluator()],
    )
    algorithms = [InContextLearning()]

    results = train_and_evaluate_benchmark(
        model=larger_model,
        tokenizer=larger_tokenizer,
        algorithms=algorithms,
        benchmark=benchmark,
        generation_config=GenerationConfig(
            max_length=100, pad_token_id=larger_tokenizer.eos_token_id
        ),
    )

    assert len(results.predictions) == 3
    assert results.predictions[0].example == test_dataset[0]
    assert results.predictions[1].example == test_dataset[1]
    assert results.predictions[2].example == test_dataset[2]
    assert results.metrics["accuracy"] == pytest.approx(2 / 3)
