from dataclasses import dataclass
import pyrallis
from pyrallis import field
from repepo.algorithms.repe import RepeReadingControl
from repepo.data.make_dataset import DatasetSpec

# TODO: Move this into main repo as it seems fairly re-used
from repepo.experiments.caa_repro.utils.helpers import (
    get_model_name,
    get_model_and_tokenizer,
)

from collections import defaultdict
from typing import Optional

import pathlib
import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# TODO: Move this config into main repo
# It's probably better than the weird Environ stuff I have going on
from repepo.experiments_2.config import WORK_DIR, DATASET_DIR
from repepo.data.make_dataset import make_dataset
from repepo.core.pipeline import Pipeline
from repepo.core.format import LlamaChatFormatter

from steering_vectors import SteeringVectorTrainingSample
from steering_vectors.layer_matching import (
    LayerType,
    ModelLayerConfig,
    guess_and_enhance_layer_config,
)
from steering_vectors.train_steering_vector import _extract_activations


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


def get_experiment_path() -> pathlib.Path:
    return WORK_DIR / "concept_vector_linearity"


@torch.no_grad()
def get_pos_and_neg_activations(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_samples: list[SteeringVectorTrainingSample] | list[tuple[str, str]],
    layers: Optional[list[int]] = None,
    layer_type: LayerType = "decoder_block",
    layer_config: Optional[ModelLayerConfig] = None,
    move_to_cpu: bool = False,
    read_token_index: int = -1,
    show_progress: bool = False,
) -> tuple[dict[int, list[Tensor]], dict[int, list[Tensor]]]:
    """
    Train a steering vector for the given model.

    Args:
        model: The model to train the steering vector for
        tokenizer: The tokenizer to use
        training_samples: A list of training samples, where each sample is a tuple of
            (positive_prompt, negative_prompt). The steering vector approximate the
            difference between the positive prompt and negative prompt activations.
        layers: A list of layer numbers to train the steering vector on. If None, train
            on all layers.
        layer_type: The type of layer to train the steering vector on. Default is
            "decoder_block".
        layer_config: A dictionary mapping layer types to layer matching functions.
            If not provided, this will be inferred automatically.
        move_to_cpu: If True, move the activations to the CPU before training. Default False.
        read_token_index: The index of the token to read the activations from. Default -1, meaning final token.
        show_progress: If True, show a progress bar. Default False.
        aggregator: A function that takes the positive and negative activations for a
            layer and returns a single vector. Default is mean_aggregator.
    """
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)
    pos_activations: dict[int, list[Tensor]] = defaultdict(list)
    neg_activations: dict[int, list[Tensor]] = defaultdict(list)
    # TODO: batching
    for i, (pos_prompt, neg_prompt) in enumerate(
        tqdm(
            training_samples, disable=not show_progress, desc="Training steering vector"
        )
    ):
        pos_acts = _extract_activations(
            model,
            tokenizer,
            pos_prompt,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_index=read_token_index,
        )
        neg_acts = _extract_activations(
            model,
            tokenizer,
            neg_prompt,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_index=read_token_index,
        )
        for layer_num, pos_act in pos_acts.items():
            if move_to_cpu:
                pos_act = pos_act.cpu()
            pos_activations[layer_num].append(pos_act)
        for layer_num, neg_act in neg_acts.items():
            if move_to_cpu:
                neg_act = neg_act.cpu()
            neg_activations[layer_num].append(neg_act)

    return pos_activations, neg_activations


def extract_concept_vectors_and_mean_relative_norms(
    config: ConceptVectorsConfig,
):
    """Run in_distribution train and eval in CAA style in a single run"""
    model_name = get_model_name(config.use_base_model, config.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)
    pipeline = Pipeline(model, tokenizer, formatter=LlamaChatFormatter())

    repe_algo = RepeReadingControl(
        patch_generation_tokens_only=True,
        # CAA reads from position -2, since the last token is ")"
        read_token_index=-2,
        # CAA skips the first generation token, so doing the same here to match
        skip_first_n_generation_tokens=1,
        verbose=config.verbose,
    )

    train_dataset = make_dataset(config.train_dataset_spec, DATASET_DIR)
    repe_training_data = repe_algo._build_steering_vector_training_data(
        train_dataset, pipeline
    )

    # Extract activations
    pos_acts, neg_acts = get_pos_and_neg_activations(
        model,
        tokenizer,
        repe_training_data,
        read_token_index=-2,
        show_progress=True,
        move_to_cpu=True,
    )

    # Calculate difference vectors
    difference_vectors: dict[int, list[Tensor]] = {}
    for layer_num in pos_acts.keys():
        difference_vectors[layer_num] = [
            pos_act - neg_act
            for pos_act, neg_act in zip(pos_acts[layer_num], neg_acts[layer_num])
        ]

    # Calculate concept vectors via mean-difference
    concept_vectors: dict[int, Tensor] = {}
    for layer_num in difference_vectors.keys():
        diff_vecs = difference_vectors[layer_num]
        concept_vec = torch.mean(torch.stack(diff_vecs), dim=0)
        concept_vectors[layer_num] = concept_vec

    # Calculate mean intra-cluster distance
    mean_relative_norms: dict[int, float] = {}
    for layer_num in difference_vectors.keys():
        concept_vector = concept_vectors[layer_num]
        concept_vector_norm = torch.norm(concept_vector)
        distances_to_mean_diff_vec = [
            torch.norm(diff_vec - concept_vector)
            for diff_vec in difference_vectors[layer_num]
        ]
        relative_distances = [
            dist / concept_vector_norm for dist in distances_to_mean_diff_vec
        ]
        mean_relative_norm = torch.mean(torch.stack(relative_distances))
        mean_relative_norms[layer_num] = mean_relative_norm.item()

    return concept_vectors, mean_relative_norms


if __name__ == "__main__":
    config = pyrallis.parse(ConceptVectorsConfig)
    (
        concept_vectors,
        mean_relative_norms,
    ) = extract_concept_vectors_and_mean_relative_norms(config)

    # Save results
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_result_save_suffix()
    vectors_save_dir = experiment_path / "vectors"
    vectors_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        concept_vectors, vectors_save_dir / f"concept_vectors_{result_save_suffix}.pt"
    )
    torch.save(
        mean_relative_norms,
        vectors_save_dir / f"mean_relative_norms_{result_save_suffix}.pt",
    )
