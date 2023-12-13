from collections import defaultdict
from typing import Any, Callable, Literal, Optional

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from repepo.core.types import Model
from repepo.utils.layer_matching import (
    LayerMatcher,
    collect_matching_layers,
    guess_decoder_block_matcher,
)
from repepo.utils.torch_utils import get_module, untuple_tensor


LayerType = Literal[
    "decoder_block", "self_attn", "mlp", "input_layernorm", "post_attention_layernorm"
]

PatchOperator = Callable[[Tensor, Tensor], Tensor]

PatchOperatorName = Literal["addition", "piecewise_addition", "projection_subtraction"]


def addition_operator(original_tensor: Tensor, patch_activation: Tensor) -> Tensor:
    return original_tensor + patch_activation


def piecewise_addition_operator(
    original_tensor: Tensor, patch_activation: Tensor
) -> Tensor:
    sign = torch.sign((original_tensor * patch_activation).sum(-1, keepdim=True))
    return original_tensor + sign * patch_activation


def projection_subtraction_operator(
    original_tensor: Tensor, patch_activation: Tensor
) -> Tensor:
    proj = (original_tensor * patch_activation).sum(-1, keepdim=True) * patch_activation
    patch_norm = patch_activation.norm()
    return original_tensor - proj / (patch_norm**2)


_NAMED_OPERATORS: dict[PatchOperatorName, PatchOperator] = {
    "addition": addition_operator,
    "piecewise_addition": piecewise_addition_operator,
    "projection_subtraction": projection_subtraction_operator,
}


ModelLayerConfig = dict[LayerType, LayerMatcher]

GptNeoxLayerConfig: ModelLayerConfig = {
    "decoder_block": "gpt_neox.layers.{num}",
    "self_attn": "gpt_neox.layers.{num}.self_attn",
    "mlp": "gpt_neox.layers.{num}.mlp",
    "input_layernorm": "gpt_neox.layers.{num}.input_layernorm",
    "post_attention_layernorm": "gpt_neox.layers.{num}.post_attention_layernorm",
}

LlamaLayerConfig: ModelLayerConfig = {
    "decoder_block": "model.layers.{num}",
    "self_attn": "model.layers.{num}.attention",
    "mlp": "model.layers.{num}.mlp",
    "input_layernorm": "model.layers.{num}.input_layernorm",
    "post_attention_layernorm": "model.layers.{num}.post_attention_layernorm",
}


class ModelPatcher:
    """Helper class to read / write model hidden activations"""

    model: Model
    config: ModelLayerConfig

    registered_hooks: dict[str, list[RemovableHandle]]

    def __init__(self, model: Model, config: Optional[ModelLayerConfig] = None):
        self.model = model
        self.config = config or {}
        self.registered_hooks = defaultdict(list)
        # TODO: guess more parts of the config if not provided
        if "decoder_block" not in self.config and (
            decoder_block_matcher := guess_decoder_block_matcher(model)
        ):
            self.config["decoder_block"] = decoder_block_matcher

    def patch_activations(
        self,
        layer_activations: dict[int, torch.Tensor],
        layer_type: LayerType = "decoder_block",
        operator: PatchOperatorName | PatchOperator = "addition",
    ) -> None:
        if isinstance(operator, str):
            operator = _NAMED_OPERATORS[operator]
        """Patch the model to add in the given activations to the given layers"""
        if layer_type not in self.config:
            raise ValueError(f"layer_type {layer_type} not provided in layer config")
        matcher = self.config[layer_type]
        matching_layers = collect_matching_layers(self.model, matcher)
        for layer_num, target_activation in layer_activations.items():
            layer_name = matching_layers[layer_num]

            # copied from set_controller, not sure why it's implemented this way
            target_activation = layer_activations[layer_num].squeeze()
            if len(target_activation.shape) == 1:
                target_activation = target_activation.reshape(1, 1, -1)

            module = get_module(self.model, layer_name)
            handle = module.register_forward_hook(
                # create the hook via function call since python only creates new scopes on functions
                _create_additive_hook(target_activation, operator)
            )
            self.registered_hooks[layer_name].append(handle)

    def remove_patches(self) -> None:
        """Remove all patches from the model"""
        for _layer_name, handles in self.registered_hooks.items():
            for handle in handles:
                handle.remove()
        self.registered_hooks = defaultdict(list)


def _create_additive_hook(
    target_activation: torch.Tensor, operator: PatchOperator
) -> Any:
    """Create a hook function that adds the given target_activation to the model output"""

    def hook_fn(_m: Any, _inputs: Any, outputs: Any) -> Any:
        original_tensor = untuple_tensor(outputs)
        original_tensor[None] = operator(original_tensor, target_activation)
        return outputs

    return hook_fn
