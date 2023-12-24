from collections import defaultdict
from typing import Any, Callable, Literal, Optional

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from repepo.core.types import Model
from repepo.utils.layer_matching import (
    LayerType,
    ModelLayerConfig,
    collect_matching_layers,
    guess_and_enhance_layer_config,
)
from repepo.utils.torch_utils import get_module, untuple_tensor

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


class ModelPatcher:
    """Helper class to read / write model hidden activations"""

    model: Model
    layer_config: ModelLayerConfig

    registered_hooks: dict[str, list[RemovableHandle]]

    def __init__(self, model: Model, layer_config: Optional[ModelLayerConfig] = None):
        self.model = model
        self.layer_config = guess_and_enhance_layer_config(model, layer_config)
        self.registered_hooks = defaultdict(list)

    def patch_activations(
        self,
        layer_activations: dict[int, torch.Tensor],
        layer_type: LayerType = "decoder_block",
        operator: PatchOperatorName | PatchOperator = "addition",
    ) -> None:
        """
        Patch the model to add in the given activations to the given layers

        Args:
            layer_activations: a dictionary mapping layer numbers to the activations to add
            layer_type: the type of layer to patch
            operator: the operator to use to combine the activations with the original activations, default addition
        """
        if isinstance(operator, str):
            operator = _NAMED_OPERATORS[operator]
        if layer_type not in self.layer_config:
            raise ValueError(f"layer_type {layer_type} not provided in layer config")
        matcher = self.layer_config[layer_type]
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
