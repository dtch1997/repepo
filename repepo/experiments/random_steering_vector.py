from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from statistics import mean
from typing import Literal, cast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
from simple_parsing import ArgumentParser
import matplotlib.pyplot as plt
from steering_vectors import SteeringVector
from repepo.core.format import LlamaChatFormatter
from repepo.core.pipeline import Pipeline
from repepo.core.types import Example, Model, Tokenizer
from repepo.steering.build_steering_training_data import (
    build_steering_vector_training_data,
)
from repepo.steering.evaluate_cross_steering import (
    CrossSteeringResult,
    evaluate_cross_steering,
)
from repepo.core.evaluate import EvalResult
from repepo.steering.plot_steering_vector_cos_similarities import (
    plot_steering_vector_cos_similarities,
)
from repepo.steering.utils.helpers import make_dataset
from steering_vectors import train_steering_vector
from repepo.data.multiple_choice.make_mwe_xrisk import make_mwe
from repepo.data.multiple_choice.make_mwe_persona import make_mwe_personas_caa
from repepo.utils.stats import bernoulli_js_dist
from repepo.experiments.persona_prompts import get_all_persona_prompts

CONFIG_SAVE_PATH = "config.json"

@dataclass
class RandomSteeringVectorExperimentConfig:
    output_dir: str
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    patch_generation_tokens_only: bool = True
    skip_first_n_generation_tokens: int = 0
    test_split: str = "50:100%"
    layer: int = 13
    multipliers: list[float] = field(
        default_factory=lambda: [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    )

@dataclass 
class RandomSteeringVectorExperimentResult:
    train_dataset_name: str
    test_dataset_names: list[str]
    steering_results: list[EvalResult]

def steer_with_random_vector(
    model: Model,
    tokenizer: Tokenizer,
    dataset_name: str,
    train_split: str,
    test_split: str,
    layer: int,
    multipliers: list[float],
    normalize_steering_magnitude_to_baseline: bool = True,
    show_progress: bool = True,
    patch_generation_tokens_only: bool = True,
    skip_first_n_generation_tokens: int = 0,
    completion_template: str | None = None,
):
    pass

def run_random_steering_vector_experiment(
    config: RandomSteeringVectorExperimentConfig,
    sge_task_id: int | None = None,
):
    make_mwe()

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.float16, device_map=0
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Save the config
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        if not os.path.exists(output_dir / CONFIG_SAVE_PATH):
            raise ValueError(
                f"Output directory {output_dir} exists but does not contain a config file."
            )
        with open(output_dir / CONFIG_SAVE_PATH, "r") as f:
            old_config_dict = json.load(f)
        old_config = RandomSteeringVectorExperimentConfig(**old_config_dict)
        if old_config != config:
            raise ValueError(
                f"Output directory {output_dir} exists but contains a different config."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / CONFIG_SAVE_PATH, "w") as f:
            json.dump(config.__dict__, f, indent=4, ensure_ascii=False)

    # Load the dataset
    all_persona_prompts = get_all_persona_prompts()
    # If task id set, only run one dataset
    if sge_task_id is not None:
        persona_prompt = list(all_persona_prompts.keys())[sge_task_id]
        all_persona_prompts = {persona_prompt: all_persona_prompts[persona_prompt]}


    for i, dataset_name in enumerate(all_persona_prompts):
        results_save_file = output_dir / f"{dataset_name}.pt"
        if results_save_file.exists():
            print(f"already ran {dataset_name}, skipping")
            continue
        print(
            f"Running experiment for dataset {dataset_name} ({i+1} / {len(all_persona_prompts)})"
        )
        result = steer_with_random_vector(
            model,
            tokenizer,
            dataset_name,
            config.train_split,
            config.test_split,
            layer=config.layer,
            normalize_steering_magnitude_to_baseline=config.normalize_steering_magnitude_to_baseline,
            patch_generation_tokens_only=config.patch_generation_tokens_only,
            skip_first_n_generation_tokens=config.skip_first_n_generation_tokens,
            completion_template=config.completion_template,
            multipliers=config.multipliers,
        )
        torch.save(result, results_save_file)
    print("Done!")