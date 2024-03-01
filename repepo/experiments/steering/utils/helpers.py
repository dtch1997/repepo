import os
import pickle
import pathlib
import torch
import pandas as pd
import gc
import hashlib

from typing import cast, Hashable, Iterable
from dataclasses import dataclass
from pyrallis import field
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import BitsAndBytesConfig
from repepo.core.types import Example
from repepo.data.make_dataset import (
    DatasetSpec,
    list_datasets as _list_dataset,
    make_dataset as _make_dataset,
)
from repepo.experiments.steering.utils.config import (
    WORK_DIR,
    LOAD_IN_4BIT,
    LOAD_IN_8BIT,
    DATASET_DIR,
)

token = os.getenv("HF_TOKEN")

SPLITS = {
    "train-dev": "0:1%",
    "train": "0:40%",
    "val": "40:50%",
    "val-dev": "40:41%",
    "test": "50:100%",
    "test-dev": "50:51%",
}


class EmptyTorchCUDACache:
    """Context manager to free GPU memory"""

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()


LayerwiseMetricsDict = dict[int, float]
LayerwiseConceptVectorsDict = dict[int, torch.Tensor]


def pretty_print_example(example: Example):
    print("Not implemented")


def get_model_name(use_base_model: bool, model_size: str):
    """Gets model name for Llama-[7b,13b], base model or chat model"""
    if use_base_model:
        model_name = f"meta-llama/Llama-2-{model_size}-hf"
    else:
        model_name = f"meta-llama/Llama-2-{model_size}-chat-hf"
    return model_name


def get_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = bool(LOAD_IN_4BIT),
    load_in_8bit: bool = bool(LOAD_IN_8BIT),
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    # Note: you must have installed 'accelerate', 'bitsandbytes' to load in 8bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=token, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


@dataclass
class ConceptVectorsConfig:
    use_base_model: bool = field(default=False)
    model_size: str = field(default="13b")
    train_dataset_name: str = field(default="sycophancy_train")
    train_split_name: str = field(default="train-dev")
    verbose: bool = True

    def make_save_suffix(self) -> str:
        str = (
            f"use-base-model={self.use_base_model}_"
            f"model-size={self.model_size}_"
            f"train-dataset={self.train_dataset_name}_"
            f"train-split={self.train_split_name}"
        )
        return hashlib.md5(str.encode()).hexdigest()


_layers = [i for i in range(10, 20)]
_multipliers = [-2 + i * 0.25 for i in range(17)]


@dataclass
class SteeringConfig(ConceptVectorsConfig):
    layers: list[int] = field(default=_layers, is_mutable=True)
    multipliers: list[float] = field(default=_multipliers, is_mutable=True)
    test_dataset_name: str = field(default="truthfulqa")
    test_split_name: str = field(default="test-dev")

    def make_steering_results_save_suffix(self) -> str:
        super_suffix = self.make_save_suffix()
        str = (
            f"{super_suffix}_"
            f"layers={self.layers}_"
            f"multipliers={self.multipliers}_"
            f"test-dataset={self.test_dataset_name}_"
            f"test-split={self.test_split_name}"
        )
        return hashlib.md5(str.encode()).hexdigest()


def get_experiment_path(
    experiment_suite: str = "concept_vector_linearity",
) -> pathlib.Path:
    return WORK_DIR / experiment_suite


def save_activation_differences(
    config: ConceptVectorsConfig, activation_differences: dict[int, list[torch.Tensor]]
):
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    activations_save_dir = experiment_path / "activations"
    activations_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        activation_differences,
        activations_save_dir / f"activation_differences_{result_save_suffix}.pt",
    )


def load_activation_differences(
    config: ConceptVectorsConfig,
) -> dict[int, list[torch.Tensor]]:
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    activations_save_dir = experiment_path / "activations"
    return torch.load(
        activations_save_dir / f"activation_differences_{result_save_suffix}.pt"
    )


def save_concept_vectors(
    config: ConceptVectorsConfig, concept_vectors: dict[int, torch.Tensor]
):
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    activations_save_dir = experiment_path / "vectors"
    activations_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        concept_vectors,
        activations_save_dir / f"concept_vectors_{result_save_suffix}.pt",
    )


def save_metrics(
    config: ConceptVectorsConfig,
    metric_name: str,
    metrics: dict[Hashable, float],
):
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    metrics_save_dir = experiment_path / "metrics"
    metrics_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        metrics,
        metrics_save_dir / f"{metric_name}_{result_save_suffix}.pt",
    )


def load_metrics(
    config: ConceptVectorsConfig,
    metric_name: str,
) -> LayerwiseMetricsDict:
    """Load layer-wise metrics for a given metric_name and config."""
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    metrics_save_dir = experiment_path / "metrics"
    return torch.load(metrics_save_dir / f"{metric_name}_{result_save_suffix}.pt")


def load_concept_vectors(
    config: ConceptVectorsConfig,
) -> LayerwiseConceptVectorsDict:
    """Load layer-wise concept vectors for a given config."""
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    activations_save_dir = experiment_path / "vectors"
    return torch.load(activations_save_dir / f"concept_vectors_{result_save_suffix}.pt")


@dataclass
class SteeringResult:
    layer_id: int
    multiplier: float
    mcq_accuracy: float
    average_key_prob: float


def save_results(
    config: SteeringConfig,
    results: list[SteeringResult],
):
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_steering_results_save_suffix()
    results_save_dir = experiment_path / "results"
    results_save_dir.mkdir(parents=True, exist_ok=True)
    with open(results_save_dir / f"results_{result_save_suffix}.pickle", "wb") as f:
        pickle.dump(results, f)


def load_results(
    config: SteeringConfig,
) -> list[SteeringResult]:
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_steering_results_save_suffix()
    results_save_dir = experiment_path / "results"
    with open(results_save_dir / f"results_{result_save_suffix}.pickle", "rb") as f:
        return cast(list[SteeringResult], pickle.load(f))


def make_dataset(
    name: str,
    split_name: str = "train",
):
    if split_name not in SPLITS:
        raise ValueError(f"Unknown split name: {split_name}")
    return _make_dataset(DatasetSpec(name=name, split=SPLITS[split_name]), DATASET_DIR)


def make_subset_of_datasets(
    subset="all",
    split_name: str = "train",
):
    datasets = list_subset_of_datasets(subset=subset)
    return [make_dataset(name, split_name) for name in datasets]


def list_subset_of_datasets(
    subset="all",
):
    if subset == "all":
        return _list_dataset(dataset_dir=DATASET_DIR)
    elif subset == "dev":
        # Selected from distinct clusters of concept vectors
        return tuple(
            [
                "believes-abortion-should-be-illegal",
                "desire-for-recursive-self-improvement",
                "truthfulqa",
                "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
                "machiavellianism",
                "desire-to-persuade-people-to-be-less-harmful-to-others",
            ]
        )
    else:
        raise ValueError(f"Unknown subset: {subset}")


def get_configs_for_datasets(
    datasets: Iterable[str],
    split_name: str = "train",
):
    return [
        ConceptVectorsConfig(train_dataset_name=dataset, train_split_name=split_name)
        for dataset in datasets
    ]


def get_steering_configs_for_datasets(
    datasets: Iterable[str],
    train_split_name: str = "train",
    test_split_name: str = "test",
):
    return [
        SteeringConfig(
            train_dataset_name=dataset,
            test_dataset_name=dataset,
            train_split_name=train_split_name,
            test_split_name=test_split_name,
        )
        for dataset in datasets
    ]


def convert_to_dataframe(example_list: list[Example]) -> pd.DataFrame:
    df = pd.DataFrame([vars(example) for example in example_list])
    return df
