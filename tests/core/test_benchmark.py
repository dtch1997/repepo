import pytest
from transformers import GenerationConfig

from repepo.algorithms.icl import InContextLearning
from repepo.core.benchmark import Benchmark, train_and_evaluate_benchmark
from repepo.core.evaluate import AccuracyEvaluator, MultipleChoiceAccuracyEvaluator
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
    # Accuracy evaluator doesn't require output probs
    for pred in results.predictions:
        assert pred.generated_output is not None
        assert pred.correct_output_probs is None
        assert pred.incorrect_outputs_probs is None
    assert results.metrics["accuracy"] == pytest.approx(2 / 3)


def test_evaluate_multiple_choice_benchmark_baseline(
    gpt2_model: Model, gpt2_tokenizer: Tokenizer
) -> None:
    dataset: Dataset = [
        Example(
            "",
            "Which country is Paris located in?",
            "France",
            incorrect_outputs=["Germany", "Italy"],
        ),
        Example(
            "",
            "Which country is Shanghai located in?",
            "China",
            incorrect_outputs=["Japan", "Thailand"],
        ),
        # giving a nonsense answer so it gets it wrong
        Example(
            "",
            "Which country is Tokyo located in?",
            "WrongAnswer",
            incorrect_outputs=["Japan"],
        ),
    ]
    benchmark = Benchmark(
        name="test benchmark",
        train_dataset=dataset,
        test_dataset=dataset,
        evaluators=[MultipleChoiceAccuracyEvaluator()],
    )
    results = train_and_evaluate_benchmark(
        model=gpt2_model,
        tokenizer=gpt2_tokenizer,
        algorithms=[],
        benchmark=benchmark,
        generation_config=GenerationConfig(
            max_length=100, pad_token_id=gpt2_tokenizer.eos_token_id
        ),
    )

    assert len(results.predictions) == 3
    assert results.predictions[0].example == dataset[0]
    assert results.predictions[1].example == dataset[1]
    assert results.predictions[2].example == dataset[2]
    # multiple choice evaluator doesn't require generation
    for pred in results.predictions:
        assert pred.generated_output is None
        assert pred.correct_output_probs is not None
        assert pred.incorrect_outputs_probs is not None
    assert len(results.predictions[0].incorrect_outputs_probs or []) == 2
    assert len(results.predictions[1].incorrect_outputs_probs or []) == 2
    assert len(results.predictions[2].incorrect_outputs_probs or []) == 1
    assert results.metrics["accuracy"] == pytest.approx(2 / 3)
