from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple

from steering_vectors import SteeringVector
from tqdm import tqdm

from repepo.core.evaluate import (
    EvalResult,
    LogitDifferenceEvaluator,
    NormalizedPositiveProbabilityEvaluator,
)
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
    dataset_baselines: list[EvalResult]
    steering: dict[float, list[list[EvalResult]]]

    @property
    def neg_steering(self) -> dict[float, list[list[EvalResult]]]:
        return {k: v for k, v in self.steering.items() if k < 0}

    @property
    def pos_steering(self) -> dict[float, list[list[EvalResult]]]:
        return {k: v for k, v in self.steering.items() if k > 0}


def evaluate_cross_steering(
    model: Model,
    tokenizer: Tokenizer,
    layer: int,
    steering_vectors: dict[str, SteeringVector],
    datasets: dict[str, Dataset],
    multipliers: list[float],
    build_pipeline: Callable[[Model, Tokenizer, str], Any] | None = None,
    patch_generation_tokens_only: bool = True,
    patch_operator: Literal["add", "ablate_then_add"] = "add",
    skip_first_n_generation_tokens: int = 0,
    completion_template: str | None = None,
    show_progress: bool = True,
) -> CrossSteeringResult:
    if build_pipeline is None:
        build_pipeline = lambda model, tokenizer, _dataset_label: Pipeline(
            model=model,
            tokenizer=tokenizer,
            formatter=LlamaChatFormatter(),
        )

    """Evaluate steering vectors on multiple datasets"""
    steering_labels = list(steering_vectors.keys())
    dataset_labels = list(datasets.keys())

    # Get baseline logits
    baseline_results = []
    steering: dict[float, list[list[EvalResult]]] = defaultdict(list)
    pbar = tqdm(
        total=len(dataset_labels) * len(steering_labels),
        desc="Evaluating cross-steering",
        disable=not show_progress,
    )

    # just need a random steering vector to get the baseline, multiplier will be 0
    first_sv = list(steering_vectors.values())[0]

    for dataset_label in dataset_labels:
        dataset_steering: dict[float, list[EvalResult]] = defaultdict(list)
        dataset = datasets[dataset_label]
        pipeline = build_pipeline(model, tokenizer, dataset_label)
        result = evaluate_steering_vector(
            pipeline,
            steering_vector=first_sv,
            dataset=dataset,
            layers=[layer],
            multipliers=[0],
            evaluators=[
                NormalizedPositiveProbabilityEvaluator(),
                LogitDifferenceEvaluator(),
            ],
            patch_operator=patch_operator,
            patch_generation_tokens_only=patch_generation_tokens_only,
            skip_first_n_generation_tokens=skip_first_n_generation_tokens,
            completion_template=completion_template,
            show_progress=False,
        )[0]
        baseline_results.append(result)
        for steering_label in steering_labels:
            steering_vector = steering_vectors[steering_label]
            results = evaluate_steering_vector(
                pipeline,
                steering_vector,
                dataset,
                layers=[layer],
                multipliers=[mul for mul in multipliers if mul != 0],
                evaluators=[
                    NormalizedPositiveProbabilityEvaluator(),
                    LogitDifferenceEvaluator(),
                ],
                patch_generation_tokens_only=patch_generation_tokens_only,
                patch_operator=patch_operator,
                skip_first_n_generation_tokens=skip_first_n_generation_tokens,
                completion_template=completion_template,
                show_progress=False,
                slim_results=True,
            )
            for result, multiplier in zip(results, multipliers):
                dataset_steering[multiplier].append(result)
            pbar.update(1)
        for multiplier, results in dataset_steering.items():
            steering[multiplier].append(results)

    return CrossSteeringResult(
        steering_labels=steering_labels,
        dataset_labels=dataset_labels,
        dataset_baselines=baseline_results,
        steering=steering,
    )
