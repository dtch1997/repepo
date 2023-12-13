import pytest
import torch
from transformers import GPTNeoXForCausalLM
from repepo.core.types import Tokenizer

from repepo.utils.model_patcher import GptNeoxLayerConfig, LayerType, ModelPatcher
from tests._original_repe.rep_control_reading_vec import WrappedReadingVecModel


@pytest.mark.parametrize(
    "layer_type",
    [
        "decoder_block",
        "mlp",
        # the other layer types appear to be broken for GPTNeoX in _original_repe
    ],
)
def test_ModelPatcher_patch_block_activations_additive_matches_WrappedReadingVecModel_set_controller(
    tokenizer: Tokenizer,
    layer_type: LayerType,
) -> None:
    # load the same model twice so we can verify that each techniques does identical things
    model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", token=True)
    model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", token=True)

    assert isinstance(model1, GPTNeoXForCausalLM)  # keep pyright happy
    assert isinstance(model2, GPTNeoXForCausalLM)  # keep pyright happy

    activations = {
        1: torch.randn(1, 512),
        -3: torch.randn(1, 512),
        -4: torch.randn(1, 512),
    }
    layers = list(activations.keys())

    model_patcher = ModelPatcher(model1, GptNeoxLayerConfig)
    reading_vec_wrapper = WrappedReadingVecModel(model2, tokenizer)
    reading_vec_wrapper.wrap_block(layers, block_name=layer_type)

    reading_vec_wrapper.set_controller(layers, activations, block_name=layer_type)
    model_patcher.patch_activations_additive(activations, layer_type=layer_type)

    inputs = tokenizer("Hello, world", return_tensors="pt")
    with torch.no_grad():
        model1_outputs = model1(**inputs, output_hidden_states=False)
        model2_outputs = model2(**inputs, output_hidden_states=False)

    # verify that the outputs are identical
    assert torch.equal(model1_outputs.logits, model2_outputs.logits)


@torch.no_grad()
def test_ModelPatcher_remove_patches_reverts_model_changes(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt")
    original_logits = model(**inputs, output_hidden_states=False).logits
    model_patcher = ModelPatcher(model, GptNeoxLayerConfig)
    model_patcher.patch_activations_additive(
        {
            1: torch.randn(1, 512),
            -1: torch.randn(1, 512),
        }
    )
    patched_logits = model(**inputs, output_hidden_states=False).logits
    model_patcher.remove_patches()
    unpatched_logits = model(**inputs, output_hidden_states=False).logits

    assert not torch.equal(original_logits, patched_logits)
    assert torch.equal(original_logits, unpatched_logits)
