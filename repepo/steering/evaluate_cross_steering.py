from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

from steering_vectors import SteeringVector
from tqdm import tqdm

from repepo.core.evaluate import EvalResult, NormalizedPositiveProbabilityEvaluator
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
    pos_steering: list[list[EvalResult]]
    neg_steering: list[list[EvalResult]]
    pos_multiplier: float
    neg_multiplier: float


def evaluate_cross_steering(
    model: Model,
    tokenizer: Tokenizer,
    layer: int,
    steering_vectors: dict[str, SteeringVector],
    datasets: dict[str, Dataset],
    build_pipeline: Callable[[Model, Tokenizer, str], Any] | None = None,
    positive_multiplier: float = 1.0,
    negative_multiplier: float = -1.0,
    patch_generation_tokens_only: bool = True,
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
    pos_steering = []
    neg_steering = []
    pbar = tqdm(
        total=len(dataset_labels) * len(steering_labels),
        desc="Evaluating cross-steering",
        disable=not show_progress,
    )

    # just need a random steering vector to get the baseline, multiplier will be 0
    first_sv = list(steering_vectors.values())[0]

    for dataset_label in dataset_labels:
        dataset_pos_steering = []
        dataset_neg_steering = []
        dataset = datasets[dataset_label]
        pipeline = build_pipeline(model, tokenizer, dataset_label)
        result = evaluate_steering_vector(
            pipeline,
            steering_vector=first_sv,
            dataset=dataset,
            layers=[layer],
            multipliers=[0.0],
            evaluators=[NormalizedPositiveProbabilityEvaluator()],
            patch_generation_tokens_only=patch_generation_tokens_only,
            skip_first_n_generation_tokens=skip_first_n_generation_tokens,
            completion_template=completion_template,
            show_progress=False,
        )[0]
        baseline_results.append(result)
        for steering_label in steering_labels:
            steering_vector = steering_vectors[steering_label]
            neg_result, pos_result = evaluate_steering_vector(
                pipeline,
                steering_vector,
                dataset,
                layers=[layer],
                multipliers=[negative_multiplier, positive_multiplier],
                patch_generation_tokens_only=patch_generation_tokens_only,
                skip_first_n_generation_tokens=skip_first_n_generation_tokens,
                completion_template=completion_template,
                show_progress=False,
            )
            dataset_neg_steering.append(neg_result)
            dataset_pos_steering.append(pos_result)
            pbar.update(1)
        pos_steering.append(dataset_pos_steering)
        neg_steering.append(dataset_neg_steering)

    return CrossSteeringResult(
        steering_labels=steering_labels,
        dataset_labels=dataset_labels,
        dataset_baselines=baseline_results,
        pos_steering=pos_steering,
        neg_steering=neg_steering,
        pos_multiplier=positive_multiplier,
        neg_multiplier=negative_multiplier,
    )
