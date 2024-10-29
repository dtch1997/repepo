from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import torch
from repepo.core.format import Llama3ChatFormatter
from repepo.steering.sweep_layers import SWEEP_DATASETS, sweep_layers
from transformers import AutoModelForCausalLM, AutoTokenizer
from repepo.data.multiple_choice.make_mwe_xrisk import make_mwe as make_mwe_xrisk_caa
from repepo.data.multiple_choice.make_mwe_persona import make_mwe_personas_caa
from repepo.data.multiple_choice.make_caa_sycophancy import make_sycophancy_caa
from repepo.data.multiple_choice.make_caa_truthfulqa import make_truthfulqa_caa
from simple_parsing import ArgumentParser



@dataclass
class LlamaSweepLayersConfig:
    output_dir: str
    layer: int
    train_split: str = "0%:10%"
    test_split: str = "20%:30%"
    multipliers: list[float] = field(
        default_factory=lambda: [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    )
    datasets: list[str] = field(default_factory=lambda: SWEEP_DATASETS)
    force: bool = False
    save_sweep_results: bool = False


def make_all_datasets():
    make_sycophancy_caa()
    make_truthfulqa_caa()
    make_mwe_xrisk_caa()
    make_mwe_personas_caa()


def sweep_llama3_70b_layers(config: LlamaSweepLayersConfig):
    print(f"Running sweep with config: {config}")
    make_all_datasets()
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-70B-Instruct", torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

    output_dir = Path(config.output_dir)
    sweep_results = sweep_layers(
        model=model,
        tokenizer=tokenizer,
        layers=[config.layer],
        train_split=config.train_split,
        test_split=config.test_split,
        datasets=config.datasets,
        multipliers=config.multipliers,
        formatter=Llama3ChatFormatter(),
        save_progress_dir=output_dir,
    )
    if output_dir is not None and config.save_sweep_results:
        torch.save(sweep_results, output_dir / "sweep_results.pt")
    return sweep_results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LlamaSweepLayersConfig, dest="config")
    args = parser.parse_args()
    config = args.config
    sweep_llama3_70b_layers(config)
