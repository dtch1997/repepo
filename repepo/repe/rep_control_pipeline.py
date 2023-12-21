from typing import Optional, cast
import torch
from transformers.pipelines import TextGenerationPipeline

from repepo.utils.model_patcher import (
    LayerType,
    ModelLayerConfig,
    ModelPatcher,
    PatchOperator,
    PatchOperatorName,
)


class RepControlPipeline(TextGenerationPipeline):
    """
    This is the RepE RepControlPipeline, but with the WrappedReadingVecModel replaced by ModelPatcher

    NOTE: This is just a temporary fix, and we should rewrite our RepE implementation to avoid any unneeded
    cruft from the original RepE repo like this class. However, we should do this replacement incrementally
    so we can ensure we don't accidentally change any behavior compared with the original implementation.
    """

    block_name: LayerType
    patch_operator: PatchOperatorName | PatchOperator

    def __init__(
        self,
        model,
        tokenizer,
        layers,
        block_name: str = "decoder_block",
        control_method="reading_vec",
        layer_config: Optional[ModelLayerConfig] = None,
        patch_operator: PatchOperatorName | PatchOperator = "addition",
        **kwargs,
    ):
        # TODO: implement different control method and supported intermediate modules for different models
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        self.model_patcher = ModelPatcher(model, layer_config)
        self.patch_operator = patch_operator
        self.block_name = cast(LayerType, block_name)
        self.layers = layers

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def __call__(self, text_inputs, activations=None, **kwargs):
        if activations is not None:
            self.model_patcher.remove_patches()
            # layers are redundant, just make sure it's not causing confusion
            assert len(self.layers) == len(activations)
            for layer in self.layers:
                assert layer in activations
            self.model_patcher.patch_activations(
                activations, self.block_name, self.patch_operator
            )

        with torch.autocast(device_type="cuda"):
            outputs = super().__call__(text_inputs, **kwargs)
        self.model_patcher.remove_patches()

        return outputs
