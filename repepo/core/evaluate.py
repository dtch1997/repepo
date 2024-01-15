# pyright: strict


from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import Optional, Sequence
from repepo.core.pipeline import TextProbs
from repepo.core.types import Example


@dataclass
class EvalPrediction:
    example: Example
    generated_output: Optional[str] = None
    correct_output_probs: Optional[TextProbs] = None
    incorrect_outputs_probs: Optional[list[TextProbs]] = None


@dataclass
class EvalResult:
    predictions: list[EvalPrediction]
    metrics: dict[str, float]


class Evaluator(ABC):
    requires_generation: bool = False
    requires_probs: bool = False

    @abstractmethod
    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        raise NotImplementedError()


class AccuracyEvaluator(Evaluator):
    """
    Evaluator that computes accuracy, i.e. the percentage of examples where the model
    generated the correct output.
    """

    requires_generation = True

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction, 1 if correct, 0 otherwise."""
        expected = prediction.example.output
        # the output might be longer than the expected depending on how many tokens we generate
        # so just verify that the expected output is a prefix of the generated output
        assert prediction.generated_output is not None, "generation is required"
        is_correct = prediction.generated_output.strip().startswith(expected.strip())
        return 1.0 if is_correct else 0.0

    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        pred_results = [self.score_prediction(pred) for pred in predictions]
        return {"accuracy": mean(pred_results)}


class MultipleChoiceAccuracyEvaluator(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the probability of
    the correct output and comparing it to the probability of the incorrect outputs.
    """

    requires_probs = True

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction, 1 if correct, 0 otherwise."""
        if (
            prediction.example.incorrect_outputs is None
            or len(prediction.example.incorrect_outputs) == 0
        ):
            raise ValueError(
                "Multiple choice evaluation requires examples to set incorrect_outputs"
            )
        # the output might be longer than the expected depending on how many tokens we generate
        # so just verify that the expected output is a prefix of the generated output
        assert prediction.correct_output_probs is not None, "output probs are required"
        assert (
            prediction.incorrect_outputs_probs is not None
        ), "output probs are required"
        correct_prob = prediction.correct_output_probs.sum_logprobs
        incorrect_probs = [
            incorrect_output_probs.sum_logprobs
            for incorrect_output_probs in prediction.incorrect_outputs_probs
        ]
        return 1.0 if correct_prob > max(incorrect_probs) else 0.0

    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        pred_results = [self.score_prediction(pred) for pred in predictions]
        return {"accuracy": mean(pred_results)}
