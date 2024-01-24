import torch

from typing import List
from repepo.algorithms.repe import RepeReadingControl
from repepo.core.pipeline import Pipeline
from repepo.core.format import LlamaChatFormatter
from repepo.data.make_dataset import make_dataset, DatasetSpec

from repepo.experiments.caa_repro.utils.helpers import (
    make_tensor_save_suffix,
    get_model_name,
    get_model_and_tokenizer,
    get_save_vectors_path,
)
from dataclasses import dataclass
import pyrallis
from pyrallis import field

save_vectors_path = get_save_vectors_path()


@torch.no_grad()
def generate_and_save_vectors(
    dataset_spec: DatasetSpec,
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
):
    # NOTE: At the moment 'layers' is not used; we instead save all layers
    # TODO: use 'layers' to only save vectors for those layers
    # TODO: Add support for saving activations
    model_name = get_model_name(use_base_model, model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)
    pipeline = Pipeline(model, tokenizer, formatter=LlamaChatFormatter())
    dataset = make_dataset(dataset_spec)

    algorithm = RepeReadingControl(skip_control=True)
    steering_vector = algorithm._get_steering_vector(pipeline, dataset)

    for layer_id, layer_activation in steering_vector.layer_activations.items():
        torch.save(
            layer_activation,
            save_vectors_path
            / f"vec_layer_{make_tensor_save_suffix(layer_id, model_name)}.pt",
        )


@dataclass
class GenerateVectorsConfig:
    layers: List[int] = field(default=[], is_mutable=True)
    save_activations: bool = False
    use_base_model: bool = False
    model_size: str = "7b"
    dataset_spec: DatasetSpec = field(
        default=DatasetSpec(name="sycophancy_train"), is_mutable=True
    )


if __name__ == "__main__":
    save_vectors_path.mkdir(parents=True, exist_ok=True)

    config = pyrallis.parse(GenerateVectorsConfig)
    generate_and_save_vectors(
        config.dataset_spec,
        config.layers,
        config.save_activations,
        config.use_base_model,
        config.model_size,
    )
