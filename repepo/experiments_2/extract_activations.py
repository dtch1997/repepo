import gc

from repepo.algorithms.repe import RepeReadingControl

# TODO: Move this into main repo as it seems fairly re-used
from repepo.experiments_2.utils.helpers import (
    get_model_name,
    get_model_and_tokenizer,
    ConceptVectorsConfig,
    save_activation_differences,
    list_datasets,
)
from repepo.experiments_2.utils.config import DATASET_DIR

from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

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
        tqdm(training_samples, disable=not show_progress, desc="Extracting activations")
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


@torch.no_grad()
def extract_difference_vectors(
    config: ConceptVectorsConfig,
):
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

    _pos, _neg = repe_training_data[0]
    print(_pos)
    print(_neg)

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

    return difference_vectors
    # # Calculate concept vectors via mean-difference
    # concept_vectors: dict[int, Tensor] = {}
    # for layer_num in difference_vectors.keys():
    #     diff_vecs = difference_vectors[layer_num]
    #     concept_vec = torch.mean(torch.stack(diff_vecs), dim=0)
    #     concept_vectors[layer_num] = concept_vec

    # # Calculate mean intra-cluster distance
    # mean_relative_norms: dict[int, float] = {}
    # for layer_num in difference_vectors.keys():
    #     concept_vector = concept_vectors[layer_num]
    #     concept_vector_norm = torch.norm(concept_vector)
    #     distances_to_mean_diff_vec = [
    #         torch.norm(diff_vec - concept_vector)
    #         for diff_vec in difference_vectors[layer_num]
    #     ]
    #     relative_distances = [
    #         dist / concept_vector_norm for dist in distances_to_mean_diff_vec
    #     ]
    #     mean_relative_norm = torch.mean(torch.stack(relative_distances))
    #     mean_relative_norms[layer_num] = mean_relative_norm.item()

    # return concept_vectors, mean_relative_norms


def run_extract_and_save(config: ConceptVectorsConfig):
    difference_vectors = extract_difference_vectors(config)
    save_activation_differences(config, difference_vectors)

    # Not sure why this is necessary, but it seems to be
    del difference_vectors
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ConceptVectorsConfig, dest="config")
    parser.add_argument("--datasets", type=str, default="")
    args = parser.parse_args()
    config = args.config

    if args.datasets:
        all_datasets = list_datasets(args.datasets)
        for dataset_name in all_datasets:
            config.train_dataset_spec.name = dataset_name
            print(f"Running on dataset: {dataset_name}")
            run_extract_and_save(config)

    else:
        run_extract_and_save(config)
