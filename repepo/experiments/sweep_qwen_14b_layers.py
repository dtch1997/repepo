from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import torch
from repepo.core.format import QwenChatFormatter
from repepo.steering.sweep_layers import sweep_layers
from transformers import AutoModelForCausalLM, AutoTokenizer
from repepo.data.multiple_choice.make_mwe_xrisk import make_mwe as make_mwe_xrisk_caa
from repepo.data.multiple_choice.make_mwe_persona import make_mwe_personas_caa
from repepo.data.multiple_choice.make_caa_sycophancy import make_sycophancy_caa
from repepo.data.multiple_choice.make_caa_truthfulqa import make_truthfulqa_caa
from simple_parsing import ArgumentParser

CONFIG_SAVE_PATH = "config.json"


@dataclass
class QwenSweepLayersConfig:
    output_dir: str
    train_split: str = "0%:10%"
    test_split: str = "50%:60%"
    multipliers: list[float] = field(
        default_factory=lambda: [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    )


def make_all_datasets():
    make_sycophancy_caa()
    make_truthfulqa_caa()
    make_mwe_xrisk_caa()
    make_mwe_personas_caa()


def sweep_qwen_14b_layers(config: QwenSweepLayersConfig):
    print(f"Running sweep with config: {config}")
    make_all_datasets()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-14B-Chat", torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")

    output_dir = Path(config.output_dir)
    if output_dir.exists():
        if not os.path.exists(output_dir / CONFIG_SAVE_PATH):
            raise ValueError(
                f"Output directory {output_dir} exists but does not contain a config file."
            )
        with open(output_dir / CONFIG_SAVE_PATH, "r") as f:
            old_config_dict = json.load(f)
        old_config = QwenSweepLayersConfig(**old_config_dict)
        if old_config != config:
            raise ValueError(
                f"Output directory {output_dir} exists but contains a different config."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / CONFIG_SAVE_PATH, "w") as f:
            json.dump(config.__dict__, f, indent=4, ensure_ascii=False)
    sweep_results = sweep_layers(
        model=model,
        tokenizer=tokenizer,
        layers=range(40),
        train_split=config.train_split,
        test_split=config.test_split,
        formatter=QwenChatFormatter(),
    )
    if output_dir is not None:
        torch.save(sweep_results, output_dir / "sweep_results.pt")
    return sweep_results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(QwenSweepLayersConfig, dest="config")
    args = parser.parse_args()
    config = args.config
    sweep_qwen_14b_layers(config)
