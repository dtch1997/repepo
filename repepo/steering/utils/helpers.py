import os
import pathlib
from typing import Literal
import torch
import pandas as pd
import gc
import json
import pickle
import hashlib

from steering_vectors.steering_vector import SteeringVector
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import BitsAndBytesConfig
from repepo.core.types import Example, Model, Tokenizer
from repepo.core.format import (
    Formatter,
    IdentityFormatter,
    LlamaChatFormatter,
    QwenChatFormatter,
)
from repepo.core.evaluate import EvalResult
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
from repepo.core.format import (
    LLAMA_7B_DEFAULT_COMPLETION_TEMPLATE,
    LLAMA_7B_DEFAULT_SYSTEM_PROMPT,
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


def get_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = bool(LOAD_IN_4BIT),
    load_in_8bit: bool = bool(LOAD_IN_8BIT),
) -> tuple[Model, Tokenizer]:
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
        return LlamaChatFormatter()
    elif formatter_name == "identity-formatter":
        return IdentityFormatter()
    elif formatter_name == "qwen-chat-formatter":
        return QwenChatFormatter()
    else:
        raise ValueError(f"Unknown formatter: {formatter_name}")


@dataclass
class SteeringConfig:
    # Steering vector training
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    train_dataset: str = "sycophancy_train"
    train_split: str = "0%:+10"
    train_system_prompt: str = LLAMA_7B_DEFAULT_SYSTEM_PROMPT
    train_completion_template: str = LLAMA_7B_DEFAULT_COMPLETION_TEMPLATE
    train_prompt_prefix: str | None = None
    test_prompt_prefix: str | None = None
    formatter: str = "llama-chat-formatter"
    aggregator: str = "mean"
    layer: int = 13
    layer_type: str = "decoder_block"
    # Steering vector evaluation
    test_dataset: str = "sycophancy_train"
    test_split: str = "40%:+10"
    test_system_prompt: str = LLAMA_7B_DEFAULT_SYSTEM_PROMPT
    test_completion_template: str = LLAMA_7B_DEFAULT_COMPLETION_TEMPLATE
    multiplier: float = 0
    patch_generation_tokens_only: bool = True
    skip_first_n_generation_tokens: int = 0
    slim_eval: bool = True
    patch_operator: Literal["add", "ablate_then_add"] = "add"

    @property
    def train_hash(self):
        hash_str = (
            f"{self.model_name}"
            f"{self.train_dataset}"
            f"{self.train_split}"
            f"{self.train_system_prompt}"
            f"{self.train_completion_template}"
            f"{self.formatter}"
            f"{self.aggregator}"
            f"{self.layer}"
            f"{self.layer_type}"
            f"{self.train_prompt_prefix}"
        )
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]

    @property
    def eval_hash(self):
        return hashlib.md5(str(self).encode()).hexdigest()[:16]


def get_experiment_path(
    experiment_suite: str = experiment_suite,
    experiment_hash: str | None = None,
    work_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    work_path = pathlib.Path(work_dir) if work_dir else WORK_DIR
    path = work_path / experiment_suite
    if experiment_hash:
        path = path / experiment_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class SteeringResult:
    config_hash: str
    # steering metrics
    mcq_acc: float
    logit_diff: float
    pos_prob: float


def get_eval_result_path(
    experiment_hash: str, work_dir: pathlib.Path | str | None = None
):
    return (
        get_experiment_path(experiment_hash=experiment_hash, work_dir=work_dir)
        / "eval_result.pickle"
    )


def save_eval_result(
    experiment_hash: str, result: EvalResult, work_dir: pathlib.Path | str | None = None
):
    # NOTE: json.dump doesn't work for nested dataclasses
    with open(get_eval_result_path(experiment_hash, work_dir=work_dir), "wb") as f:
        pickle.dump(result, f)


def load_eval_result(
    experiment_hash: str, work_dir: pathlib.Path | str | None = None
) -> EvalResult:
    # NOTE: json.load doesn't work for nested dataclasses
    with open(get_eval_result_path(experiment_hash, work_dir=work_dir), "rb") as f:
        return pickle.load(f)


def get_activation_path(
    experiment_hash: str, work_dir: pathlib.Path | str | None = None
) -> pathlib.Path:
    experiment_path = get_experiment_path(
        experiment_hash=experiment_hash, work_dir=work_dir
    )
    return experiment_path / "activations.pt"


def save_activation(
    experiment_hash: str,
    pos_acts: list[torch.Tensor],
    neg_acts: list[torch.Tensor],
    work_dir: pathlib.Path | str | None = None,
):
    torch.save(
        {
            "pos_acts": pos_acts,
            "neg_acts": neg_acts,
        },
        get_activation_path(experiment_hash, work_dir=work_dir),
    )


def load_activation(
    experiment_hash: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    tensor_dict = torch.load(get_activation_path(experiment_hash))
    return tensor_dict["pos_acts"], tensor_dict["neg_acts"]


def get_steering_vector_path(
    experiment_hash: str, work_dir: pathlib.Path | str | None = None
) -> pathlib.Path:
    experiment_path = get_experiment_path(
        experiment_hash=experiment_hash, work_dir=work_dir
    )
    return experiment_path / "steering_vector.pt"


def save_steering_vector(
    experiment_hash: str,
    steering_vector: SteeringVector,
    work_dir: pathlib.Path | str | None = None,
):
    torch.save(
        steering_vector,
        get_steering_vector_path(experiment_hash, work_dir=work_dir),
    )


def load_steering_vector(
    experiment_hash: str,
    work_dir: pathlib.Path | str | None = None,
) -> SteeringVector:
    return torch.load(get_steering_vector_path(experiment_hash, work_dir=work_dir))


def get_metric_path(
    experiment_hash: str, metric_name: str, work_dir: pathlib.Path | str | None = None
) -> pathlib.Path:
    experiment_path = get_experiment_path(
        experiment_hash=experiment_hash, work_dir=work_dir
    )
    return experiment_path / f"metric_{metric_name}.json"


def save_metric(
    experiment_hash: str,
    metric_name: str,
    metric: float,
    work_dir: pathlib.Path | str | None = None,
):
    with open(
        get_metric_path(experiment_hash, metric_name, work_dir=work_dir), "w"
    ) as f:
        json.dump(metric, f)


def load_metric(
    experiment_hash: str, metric_name: str, work_dir: pathlib.Path | str | None = None
) -> float:
    with open(
        get_metric_path(experiment_hash, metric_name, work_dir=work_dir), "r"
    ) as f:
        return float(json.load(f))


def make_dataset(name: str, split: str = "0%:100%"):
    return _make_dataset(DatasetSpec(name=name, split=split), DATASET_DIR)


def convert_to_dataframe(example_list: list[Example]) -> pd.DataFrame:
    df = pd.DataFrame([vars(example) for example in example_list])
    return df
