from collections import defaultdict
from typing import NamedTuple, Optional
import torch
from transformers import PreTrainedTokenizerBase
from torch import nn, Tensor

from .steering_vector import SteeringVector
from .layer_matching import ModelLayerConfig, LayerType, guess_and_enhance_layer_config
from .record_activations import record_activations


class SteeringVectorTrainingSample(NamedTuple):
    positive_prompt: str
    negative_prompt: str


@torch.no_grad()
def train_steering_vector(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_samples: list[SteeringVectorTrainingSample] | list[tuple[str, str]],
    layers: Optional[list[int]] = None,
    layer_type: LayerType = "decoder_block",
    layer_config: Optional[ModelLayerConfig] = None,
    move_to_cpu: bool = False,
    read_token_index: int = -1,
    # TODO: add more options to control training
) -> SteeringVector:
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)
    pos_activations = defaultdict(list)
    neg_activations = defaultdict(list)
    # TODO: batching
    for pos_prompt, neg_prompt in training_samples:
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
    layer_activations = {}
    for layer_num in pos_activations.keys():
        pos_acts = pos_activations[layer_num]
        neg_acts = neg_activations[layer_num]
        # TODO: allow controlling how to combine activations
        direction_vec = (torch.stack(pos_acts) - torch.stack(neg_acts)).mean(dim=0)
        layer_activations[layer_num] = direction_vec
    return SteeringVector(layer_activations, layer_type)


def _extract_activations(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    layer_type: LayerType,
    layer_config: ModelLayerConfig,
    layers: list[int] | None,
    read_token_index: int,
) -> dict[int, Tensor]:
    input = tokenizer(prompt, return_tensors="pt").to(model.device)
    results = {}
    with record_activations(
        model, layer_type, layer_config, layer_nums=layers
    ) as record:
        model(**input)
    for layer_num, activation in record.items():
        results[layer_num] = activation[-1][0, read_token_index].detach()
    return results
