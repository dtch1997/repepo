# pyright: strict


from dataclasses import dataclass
from statistics import mean
from typing import Callable, Sequence
from repepo.core.types import Example


@dataclass
class EvalPrediction:
    example: Example
    output: str


@dataclass
class EvalResult:
    predictions: list[EvalPrediction]
    metrics: dict[str, float]


Evaluator = Callable[[Sequence[EvalPrediction]], dict[str, float]]


class AccuracyEvaluator:
    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction, 1 if correct, 0 otherwise."""
        expected = prediction.example.output
        # the output might be longer than the expected depending on how many tokens we generate
        # so just verify that the expected output is a prefix of the generated output
        is_correct = prediction.output.strip().startswith(expected.strip())
        return 1.0 if is_correct else 0.0

    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        pred_results = [self.score_prediction(pred) for pred in predictions]
        return {"accuracy": mean(pred_results)}
