import os
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from repepo.variables import Environ
from dataclasses import dataclass
from pyrallis import field
from repepo.data.make_dataset import DatasetSpec
from typing import Optional

token = os.getenv("HF_TOKEN")


@dataclass
class SteeringSettings:
    """
    max_new_tokens: Maximum number of tokens to generate.
    type: Type of test to run. One of "in_distribution", "out_of_distribution", "truthful_qa".
    few_shot: Whether to test with few-shot examples in the prompt. One of "positive", "negative", "none".
    do_projection: Whether to project activations onto orthogonal complement of steering vector.
    override_vector: If not None, the steering vector generated from this layer's activations will be used at all layers. Use to test the effect of steering with a different layer's vector.
    override_vector_model: If not None, steering vectors generated from this model will be used instead of the model being used for generation - use to test vector transference between models.
    use_base_model: Whether to use the base model instead of the chat model.
    model_size: Size of the model to use. One of "7b", "13b".
    n_test_datapoints: Number of datapoints to test on. If None, all datapoints will be used.
    add_every_token_position: Whether to add steering vector to every token position (including question), not only to token positions corresponding to the model's response to the user
    override_model_weights_path: If not None, the model weights at this path will be used instead of the model being used for generation - use to test using activation steering on top of fine-tuned model.
    """

    dataset_spec: DatasetSpec = field(
        default=DatasetSpec(name="sycophancy_test"), is_mutable=True
    )
    max_new_tokens: int = 100
    type: str = "in_distribution"
    few_shot: str = "none"
    do_projection: bool = False
    override_vector: Optional[int] = None
    override_vector_model: Optional[str] = None
    use_base_model: bool = False
    model_size: str = "7b"
    n_test_datapoints: Optional[int] = None
    add_every_token_position: bool = False
    override_model_weights_path: Optional[str] = None

    def make_result_save_suffix(
        self,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            "layer": layer,
            "multiplier": multiplier,
            "max_new_tokens": self.max_new_tokens,
            "type": self.type,
            "few_shot": self.few_shot,
            "do_projection": self.do_projection,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "use_base_model": self.use_base_model,
            "model_size": self.model_size,
            "n_test_datapoints": self.n_test_datapoints,
            "add_every_token_position": self.add_every_token_position,
            "override_model_weights_path": self.override_model_weights_path,
        }
        return "_".join(
            [
                f"{k}={str(v).replace('/', '-')}"
                for k, v in elements.items()
                if v is not None
            ]
        )

    def filter_result_files_by_suffix(
        self,
        directory: str,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            "layer": layer,
            "multiplier": multiplier,
            "max_new_tokens": self.max_new_tokens,
            "type": self.type,
            "few_shot": self.few_shot,
            "do_projection": self.do_projection,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "use_base_model": self.use_base_model,
            "model_size": self.model_size,
            "n_test_datapoints": self.n_test_datapoints,
            "add_every_token_position": self.add_every_token_position,
            "override_model_weights_path": self.override_model_weights_path,
        }

        filtered_elements = {k: v for k, v in elements.items() if v is not None}
        remove_elements = {k for k, v in elements.items() if v is None}

        matching_files = []

        print(self.override_model_weights_path)

        for filename in os.listdir(directory):
            if all(
                f"{k}={str(v).replace('/', '-')}" in filename
                for k, v in filtered_elements.items()
            ):
                # ensure remove_elements are *not* present
                if all(f"{k}=" not in filename for k in remove_elements):
                    matching_files.append(filename)

        return [os.path.join(directory, f) for f in matching_files]


def get_save_vectors_path():
    return (
        pathlib.Path(Environ.ProjectDir).absolute()
        / "repepo"
        / "experiments"
        / "caa_repro"
        / "vectors"
    )


def get_experiment_path(experiment_name: str = "caa_repro"):
    return pathlib.Path(Environ.ProjectDir) / "repepo" / "experiments" / experiment_name


def make_tensor_save_suffix(layer: int, model_name_path: str):
    return f'{layer}_{model_name_path.split("/")[-1]}'


def get_model_name(use_base_model: bool, model_size: str):
    """Gets model name for Llama-[7b,13b], base model or chat model"""
    if use_base_model:
        model_name = f"meta-llama/Llama-2-{model_size}-hf"
    else:
        model_name = f"meta-llama/Llama-2-{model_size}-chat-hf"
    return model_name


def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    # Note: you must have installed 'accelerate', 'bitsandbytes' to load in 8bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name, use_auth_token=token, load_in_8bit=True
    )
    return model, tokenizer
