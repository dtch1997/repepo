import os
import pathlib

from dataclasses import dataclass
from pyrallis import field
from transformers import AutoModelForCausalLM, AutoTokenizer

from repepo.core.types import Example
from repepo.data.make_dataset import DatasetSpec, make_dataset
from repepo.experiments_2.utils.config import (
    WORK_DIR,
    LOAD_IN_4BIT,
    LOAD_IN_8BIT,
)

token = os.getenv("HF_TOKEN")

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    # Note: you must have installed 'accelerate', 'bitsandbytes' to load in 8bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=token, 
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit
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

    def make_result_save_suffix(self) -> str:
        return f"use-base-model={self.use_base_model}_model-size={self.model_size}_dataset={self.train_dataset_spec}"

def get_experiment_path(
    experiment_suite: str = "concept_vector_linearity"
) -> pathlib.Path:
    return WORK_DIR / experiment_suite