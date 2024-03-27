from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

from steering_vectors import SteeringVector

from repepo.core.evaluate import NormalizedPositiveProbabilityEvaluator, evaluate
from repepo.core.format import LlamaChatFormatter
from repepo.core.pipeline import Pipeline
from repepo.core.types import Dataset, Model, Tokenizer
from repepo.steering.evaluate_steering_vector import evaluate_steering_vector


class MetricVal(NamedTuple):
    mean: float
    std: float


@dataclass
class CrossSteeringResult:
    steering_labels: list[str]
    dataset_labels: list[str]
    dataset_baseline: list[MetricVal]
    pos_steering: list[list[MetricVal]]
    neg_steering: list[list[MetricVal]]


def evaluate_cross_steering(
    model: Model,
    tokenizer: Tokenizer,
    layer: int,
    steering_vectors: dict[str, SteeringVector],
    datasets: dict[str, Dataset],
    build_pipeline: Callable[[Model, Tokenizer, str], Any] | None = None,
    positive_multiplier: float = 1.0,
    negative_multiplier: float = -1.0,
) -> CrossSteeringResult:
    if build_pipeline is None:
        build_pipeline = lambda model, tokenizer, _dataset_label: Pipeline(
            model=model, tokenizer=tokenizer, formatter=LlamaChatFormatter()
        )

    """Evaluate steering vectors on multiple datasets"""
    steering_labels = list(steering_vectors.keys())
    dataset_labels = list(datasets.keys())

    # Get baseline logits
    baseline_results = []
    pos_steering = []
    neg_steering = []
    for dataset_label in dataset_labels:
        dataset_pos_steering = []
        dataset_neg_steering = []
        dataset = datasets[dataset_label]
        pipeline = build_pipeline(model, tokenizer, dataset_label)
        result = evaluate(
            pipeline,
            dataset,
            evaluators=[NormalizedPositiveProbabilityEvaluator()],
        )
        baseline_results.append(
            MetricVal(result.metrics["mean_pos_prob"], result.metrics["std_pos_prob"])
        )
        for steering_label in steering_labels:
            steering_vector = steering_vectors[steering_label]
            neg_result, pos_result = evaluate_steering_vector(
                pipeline,
                steering_vector,
                dataset,
                layers=[layer],
                multipliers=[negative_multiplier, positive_multiplier],
                evaluators=[NormalizedPositiveProbabilityEvaluator()],
            )
            dataset_neg_steering.append(
                MetricVal(
                    neg_result.metrics["mean_pos_prob"],
                    neg_result.metrics["std_pos_prob"],
                )
            )
            dataset_pos_steering.append(
                MetricVal(
                    pos_result.metrics["mean_pos_prob"],
                    pos_result.metrics["std_pos_prob"],
                )
            )
        pos_steering.append(dataset_pos_steering)
        neg_steering.append(dataset_neg_steering)

    return CrossSteeringResult(
        steering_labels=steering_labels,
        dataset_labels=dataset_labels,
        dataset_baseline=baseline_results,
        pos_steering=pos_steering,
        neg_steering=neg_steering,
    )
