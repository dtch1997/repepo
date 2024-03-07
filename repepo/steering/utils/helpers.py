import os
import pickle
import pathlib
import torch
import pandas as pd
import gc
import hashlib

from typing import cast, Literal
from dataclasses import dataclass
from pyrallis import field
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import BitsAndBytesConfig
from repepo.core.types import Example
from repepo.core.format import Formatter
from repepo.data.make_dataset import (
    DatasetSpec,
    make_dataset as _make_dataset,
)
from repepo.steering.utils.variables import (
    WORK_DIR,
    LOAD_IN_4BIT,
    LOAD_IN_8BIT,
    DATASET_DIR,
)

token = os.getenv("HF_TOKEN")
experiment_suite = os.getenv("EXPERIMENT_SUITE", "steering-vectors")

SPLITS = {
    "train": "0%:40%",
    "val": "40%:50%",
    "test": "50:100%",
    # For development, use 10 examples per split
    "train-dev": "0%:+10",
    "val-dev": "40%:+10",
    "test-dev": "50:+10%",
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


def get_formatter(
    formatter_name: str = "llama-chat-formatter",
) -> Formatter:
    if formatter_name == "llama-chat-formatter":
        from repepo.core.format import LlamaChatFormatter

        return LlamaChatFormatter()
    elif formatter_name == "identity-formatter":
        from repepo.core.format import IdentityFormatter

        return IdentityFormatter()
    else:
        raise ValueError(f"Unknown formatter: {formatter_name}")


_layers = [0, 15, 31]  # only the first, middle, and last layer of llama-7b
_multipliers = [-1, -0.5, 0, 0.5, 1]


@dataclass
class SteeringConfig:
    use_base_model: bool = field(default=False)
    model_size: str = field(default="7b")
    train_dataset_name: str = field(default="sycophancy_train")
    train_split_name: str = field(default="train-dev")
    formatter: str = field(default="llama-chat-formatter")
    aggregator: str = field(default="mean")
    layers: list[int] = field(default=_layers, is_mutable=True)
    multipliers: list[float] = field(default=_multipliers, is_mutable=True)
    test_dataset_name: str = field(default="sycophancy_train")
    test_split_name: str = field(default="val-dev")
    test_completion_template: str = field(default="{prompt} {response}")
    patch_generation_tokens_only: bool = field(default=True)
    skip_first_n_generation_tokens: int = field(default=0)

    def make_save_suffix(self) -> str:
        # TODO: any way to loop over fields instead of hardcoding?
        str = (
            f"use-base-model={self.use_base_model}_"
            f"model-size={self.model_size}_"
            f"formatter={self.formatter}_"
            f"train-dataset={self.train_dataset_name}_"
            f"train-split={self.train_split_name}_"
            f"aggregator={self.aggregator}"
            f"layers={self.layers}_"
            f"multipliers={self.multipliers}_"
            f"test-dataset={self.test_dataset_name}_"
            f"test-split={self.test_split_name}_"
            f"test-completion-template={self.test_completion_template}_"
            f"patch-generation-tokens-only={self.patch_generation_tokens_only}_"
            f"skip-first-n-generation-tokens={self.skip_first_n_generation_tokens}"
        )
        return hashlib.md5(str.encode()).hexdigest()


def get_experiment_path(
    experiment_suite: str = experiment_suite,
) -> pathlib.Path:
    return WORK_DIR / experiment_suite


ActivationType = Literal["positive", "negative", "difference"]


def save_activations(
    config: SteeringConfig,
    activations: dict[int, list[torch.Tensor]],
    activation_type: ActivationType = "difference",
):
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    activations_save_dir = experiment_path / "activations"
    activations_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        activations,
        activations_save_dir / f"activations_{activation_type}_{result_save_suffix}.pt",
    )


def load_activations(
    config: SteeringConfig, activation_type: ActivationType = "difference"
) -> dict[int, list[torch.Tensor]]:
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    activations_save_dir = experiment_path / "activations"
    return torch.load(
        activations_save_dir / f"activations_{activation_type}_{result_save_suffix}.pt"
    )


def save_concept_vectors(
    config: SteeringConfig, concept_vectors: dict[int, torch.Tensor]
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
    config: SteeringConfig,
    metric_name: str,
    metrics: dict[int, float],
):
    """Save layer-wise metrics for a given config."""
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    metrics_save_dir = experiment_path / "metrics"
    metrics_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        metrics,
        metrics_save_dir / f"{metric_name}_{result_save_suffix}.pt",
    )


def load_metrics(
    config: SteeringConfig,
    metric_name: str,
) -> dict[int, dict[str, float]]:
    """Load layer-wise metrics for a given config."""
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    metrics_save_dir = experiment_path / "metrics"
    return torch.load(metrics_save_dir / f"{metric_name}_{result_save_suffix}.pt")


def load_concept_vectors(
    config: SteeringConfig,
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
    # steering vector metrics
    mcq_acc: float
    logit_diff: float
    pos_prob: float
    # 'conceptness' metrics?


def save_results(
    config: SteeringConfig,
    results: list[SteeringResult],
):
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    results_save_dir = experiment_path / "results"
    results_save_dir.mkdir(parents=True, exist_ok=True)
    with open(results_save_dir / f"results_{result_save_suffix}.pickle", "wb") as f:
        pickle.dump(results, f)


def get_results_path(config: SteeringConfig) -> pathlib.Path:
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
    results_save_dir = experiment_path / "results"
    return results_save_dir / f"results_{result_save_suffix}.pickle"


def load_results(
    config: SteeringConfig,
) -> list[SteeringResult]:
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_save_suffix()
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


def convert_to_dataframe(example_list: list[Example]) -> pd.DataFrame:
    df = pd.DataFrame([vars(example) for example in example_list])
    return df
