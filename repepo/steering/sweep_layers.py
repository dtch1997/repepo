from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from steering_vectors import SteeringVector, train_steering_vector
import torch
from tqdm import tqdm
from repepo.core.evaluate import (
    EvalResult,
    LogitDifferenceEvaluator,
    NormalizedPositiveProbabilityEvaluator,
)
from repepo.core.format import Formatter, QwenChatFormatter
from repepo.core.pipeline import Pipeline
from repepo.core.types import Model, Tokenizer
from repepo.steering.build_steering_training_data import (
    build_steering_vector_training_data,
)
from repepo.steering.evaluate_steering_vector import evaluate_steering_vector
from repepo.steering.steerability import get_steerability_slope
from repepo.steering.utils.helpers import (
    make_dataset,
)


@dataclass
class SweepLayersResult:
    steering_vectors: dict[str, dict[int, SteeringVector]]
    multipliers: list[float]
    layers: list[int]
    steering_results: dict[str, dict[int, list[EvalResult]]]

    @property
    def steerabilities(self) -> dict[str, dict[int, float]]:
        multipliers = np.array(self.multipliers)
        steerabilities: dict[str, dict[int, float]] = defaultdict(dict)
        for dataset, layer_results in self.steering_results.items():
            for layer, multiplier_results in layer_results.items():
                propensities = [
                    result.metrics["mean_logit_diff"] for result in multiplier_results
                ]
                propensities_arr = np.array([propensities])
                steerability = get_steerability_slope(multipliers, propensities_arr)[
                    0
                ].item()
                steerabilities[dataset][layer] = steerability
        return steerabilities


SWEEP_DATASETS = [
    "anti-immigration",
    "believes-abortion-should-be-illegal",
    "conscientiousness",
    "desire-for-acquiring-compute",
    "risk-seeking",
    "openness",
    "self-replication",
    "very-small-harm-justifies-very-large-benefit",
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
]


def train_steering_vectors_for_sweep(
    model: Model,
    tokenizer: Tokenizer,
    pipeline: Pipeline,
    datasets: list[str],
    train_split: str,
    layers: Sequence[int],
    show_progress: bool = True,
    save_progress_dir: Path | None = None,
    force: bool = False,
) -> dict[str, dict[int, SteeringVector]]:
    steering_vectors: dict[str, dict[int, SteeringVector]] = defaultdict(dict)
    pipeline = Pipeline(model, tokenizer, formatter=QwenChatFormatter())
    pbar = tqdm(
        total=len(datasets) * len(layers),
        desc="Training steering vectors",
        disable=not show_progress,
    )
    for dataset in datasets:
        train_dataset = make_dataset(dataset, train_split)
        for layer in layers:
            save_name = f"sv_{dataset}_{layer}.pt"
            if (
                save_progress_dir is not None
                and (save_progress_dir / save_name).exists()
                and not force
            ):
                steering_vector = torch.load(save_progress_dir / save_name)
                steering_vectors[dataset][layer] = steering_vector
                pbar.update(1)
                continue
            steering_vector_training_data = build_steering_vector_training_data(
                pipeline, train_dataset
            )
            steering_vector = train_steering_vector(
                model,
                tokenizer,
                steering_vector_training_data,
                layers=[layer],
                show_progress=False,
            )
            if save_progress_dir is not None:
                torch.save(steering_vector, save_progress_dir / save_name)
            steering_vectors[dataset][layer] = steering_vector
            pbar.update(1)
    return steering_vectors


def sweep_layers(
    model: Model,
    tokenizer: Tokenizer,
    formatter: Formatter,
    layers: Sequence[int],
    train_split: str = "0%:50%",
    test_split: str = "50%:100%",
    datasets: list[str] = SWEEP_DATASETS,
    multipliers: Iterable[float] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
    show_progress: bool = True,
    save_progress_dir: Path | None = None,
    force: bool = False,
) -> SweepLayersResult:
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    # steering_vectors = train_steering_vectors_for_sweep(
    #     model,
    #     tokenizer,
    #     pipeline,
    #     datasets=datasets,
    #     train_split=train_split,
    #     layers=layers,
    #     show_progress=show_progress,
    #     save_progress_dir=save_progress_dir,
    #     force=force,
    # )
    steering_vectors = {}
    steering_results: dict[str, dict[int, list[EvalResult]]] = defaultdict(dict)
    pbar = tqdm(
        total=len(datasets) * len(layers),
        desc="Evaluating steering",
        disable=not show_progress,
    )
    for dataset in datasets:
        test_dataset = make_dataset(dataset, test_split)
        for layer in layers:
            save_name = f"multiplier_res_{dataset}_{layer}.pt"
            if (
                save_progress_dir is not None
                and (save_progress_dir / save_name).exists()
                and not force
            ):
                multiplier_results = torch.load(save_progress_dir / save_name)
                steering_results[dataset][layer] = multiplier_results
                pbar.update(1)
                continue
            steering_vector = torch.load(save_progress_dir / f"sv_{dataset}_{layer}.pt")
            multiplier_results = evaluate_steering_vector(
                pipeline,
                steering_vector,
                test_dataset,
                layers=[layer],
                multipliers=list(multipliers),
                evaluators=[
                    NormalizedPositiveProbabilityEvaluator(),
                    LogitDifferenceEvaluator(),
                ],
                show_progress=False,
                slim_results=True,
            )
            steering_results[dataset][layer] = multiplier_results
            if save_progress_dir is not None:
                torch.save(multiplier_results, save_progress_dir / save_name)
            pbar.update(1)
    return SweepLayersResult(
        steering_vectors=steering_vectors,
        multipliers=list(multipliers),
        layers=list(layers),
        steering_results=steering_results,
    )
