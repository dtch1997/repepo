from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

from torch import nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .layer_matching import (
    LayerType,
    ModelLayerConfig,
    guess_and_enhance_layer_config,
    collect_matching_layers,
)
from .torch_utils import get_module, untuple_tensor


PatchOperator = Callable[[Tensor, Tensor], Tensor]


def addition_operator(original_tensor: Tensor, patch_activation: Tensor) -> Tensor:
    return original_tensor + patch_activation


@dataclass
class SteeringPatchHandle:
    model_hooks: list[RemovableHandle]

    def remove(self) -> None:
        for hook in self.model_hooks:
            hook.remove()


@dataclass
class SteeringVector:
    # activations are expected to have only 1 dimension
    layer_activations: dict[int, Tensor]
    layer_type: LayerType = "decoder_block"

    def patch_activations(
        self,
        model: nn.Module,
        layer_config: Optional[ModelLayerConfig] = None,
        operator: PatchOperator = addition_operator,
        multiplier: float = 1.0,
    ) -> SteeringPatchHandle:
        """
        Patch the activations of the given model with this steering vector.
        This will modify the model in-place, and return a handle that can be used to undo the patching.
        To automatically undo the patching, use the `apply` context manager.

        Args:
            model: The model to patch
            layer_config: A dictionary mapping layer types to layer matching functions.
                If not provided, this will be inferred automatically.
            operator: A function that takes the original activation and the target activation
                and returns the new activation. Default is addition.
            multiplier: A multiplier to scale the patch activations. Default is 1.0.
        Example:
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
            >>> steering_vector = SteeringVector(...)
            >>> handle = steering_vector.patch_activations(model)
            >>> model.forward(...)
            >>> handle.remove()
        """
        layer_config = guess_and_enhance_layer_config(
            model, layer_config, self.layer_type
        )
        hooks: list[RemovableHandle] = []
        if self.layer_type not in layer_config:
            raise ValueError(
                f"layer_type {self.layer_type} not provided in layer config"
            )
        matcher = layer_config[self.layer_type]
        matching_layers = collect_matching_layers(model, matcher)
        for layer_num, target_activation in self.layer_activations.items():
            layer_name = matching_layers[layer_num]

            target_activation = multiplier * self.layer_activations[layer_num]
            if len(target_activation.shape) == 1:
                target_activation = target_activation.reshape(1, 1, -1)

            module = get_module(model, layer_name)
            handle = module.register_forward_hook(
                # create the hook via function call since python only creates new scopes on functions
                _create_patch_hook(target_activation, operator)
            )
            hooks.append(handle)
        return SteeringPatchHandle(hooks)

    @contextmanager
    def apply(
        self,
        model: nn.Module,
        layer_config: Optional[ModelLayerConfig] = None,
        operator: PatchOperator = addition_operator,
        multiplier: float = 1.0,
    ):
        """
        Apply this steering vector to the given model.

        Args:
            model: The model to patch
            layer_config: A dictionary mapping layer types to layer matching functions.
                If not provided, this will be inferred automatically.
            operator: A function that takes the original activation and the target activation
                and returns the new activation. Default is addition.
            multiplier: A multiplier to scale the patch activations. Default is 1.0.
        Example:
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
            >>> steering_vector = SteeringVector(...)
            >>> with steering_vector.apply(model):
            >>>     model.forward(...)
        """
        try:
            handle = self.patch_activations(
                model=model,
                layer_config=layer_config,
                operator=operator,
                multiplier=multiplier,
            )
            yield
        finally:
            handle.remove()


def _create_patch_hook(target_activation: Tensor, operator: PatchOperator) -> Any:
    """Create a hook function that adds the given target_activation to the model output"""

    def hook_fn(_m: Any, _inputs: Any, outputs: Any) -> Any:
        original_tensor = untuple_tensor(outputs)
        act = target_activation.to(original_tensor.device)
        original_tensor[None] = operator(original_tensor, act)
        return outputs

    return hook_fn
