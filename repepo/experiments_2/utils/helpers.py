import os
import pathlib
import torch
import pandas as pd

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
from repepo.experiments_2.utils.config import (
    WORK_DIR,
    LOAD_IN_4BIT,
    LOAD_IN_8BIT,
    DATASET_DIR,
)

token = os.getenv("HF_TOKEN")

SPLITS = {
    "train": ":40%",
    "val": "40:50%",
    "test": "50:100%",
}

def pretty_print_example(example: Example):
    print("Example(")
    print("\tinstruction=", example.instruction)
    print("\tinput=", example.input)
    print("\tcorrect_output=", example.output)
    print("\tincorrect_output=", example.incorrect_outputs)
    print("\tmeta=", example.meta)
    print(")")


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
    train_dataset_spec: DatasetSpec = field(
        default=DatasetSpec(name="subscribes-to-virtue-ethics"), is_mutable=True
    )
    verbose: bool = True

    def make_save_suffix(self) -> str:
        return f"use-base-model={self.use_base_model}_model-size={self.model_size}_dataset={self.train_dataset_spec}"

@dataclass 
class SteeringConfig(ConceptVectorsConfig):
    layers: list[int] = field(
        default=[], 
        is_mutable=True
    )
    multipliers: list[float] = field(
        default=[-1, 0, 1], 
        is_mutable=True
    )
    test_dataset_spec: DatasetSpec = field(
        default=DatasetSpec(name="subscribes-to-virtue-ethics"), is_mutable=True
    )

    def make_steering_results_save_suffix(self) -> str:
        super_suffix = self.make_save_suffix()
        return f"{super_suffix}_layers={self.layers}_multipliers={self.multipliers}_test-dataset={self.test_dataset_spec}"

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
    config: ConceptVectorsConfig, 
    concept_vectors: dict[int, torch.Tensor]
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
) -> dict[str, float]:
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    metrics_save_dir = experiment_path / "metrics"
    return torch.load(
        metrics_save_dir / f"{metric_name}_{result_save_suffix}.pt"
    )


def load_concept_vectors(
    config: ConceptVectorsConfig,
) -> dict[int, torch.Tensor]:
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    activations_save_dir = experiment_path / "vectors"
    return torch.load(
        activations_save_dir / f"concept_vectors_{result_save_suffix}.pt"
    )

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
    torch.save(
        results,
        results_save_dir / f"results_{result_save_suffix}.pt",
    )

def load_results(
    config: SteeringConfig,
) -> list[SteeringResult]:
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_steering_results_save_suffix()
    results_save_dir = experiment_path / "results"
    return torch.load(
        results_save_dir / f"results_{result_save_suffix}.pt"
    )

def make_dataset(
    name: str,
    split_name: str = "train",
):
    if not split_name in SPLITS:
        raise ValueError(f"Unknown split name: {split_name}")
    return _make_dataset(DatasetSpec(name=name, split=SPLITS[split_name]))

def make_subset_of_datasets(
    subset = "all",
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
                f"believes-abortion-should-be-illegal",
                f"desire-for-recursive-self-improvement",
                f"truthfulqa",
                f"willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
                f"machiavellianism",
                f"desire-to-persuade-people-to-be-less-harmful-to-others",
            ]
        )
    else:
        raise ValueError(f"Unknown subset: {subset}")

def get_configs_for_datasets(
    datasets: Iterable[str],
    split: str = ":100%",
):
    return [ConceptVectorsConfig(train_dataset_spec=DatasetSpec(name=dataset, split=split)) for dataset in datasets]

def convert_to_dataframe(example_list: list[Example]) -> pd.DataFrame:
    df = pd.DataFrame([vars(example) for example in example_list])
    return df
