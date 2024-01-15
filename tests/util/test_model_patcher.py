import pytest
import torch

from transformers.models.gpt_neox import GPTNeoXForCausalLM
from repepo.core.types import Tokenizer
from repepo.utils.layer_matching import GptNeoxLayerConfig

from repepo.utils.model_patcher import (
    LayerType,
    ModelPatcher,
)
from tests._original_repe.rep_control_reading_vec import WrappedReadingVecModel


@pytest.mark.parametrize(
    "layer_type",
    [
        "decoder_block",
        "mlp",
        # the other layer types appear to be broken for GPTNeoX in _original_repe
    ],
)
def test_ModelPatcher_patch_activations_matches_WrappedReadingVecModel_set_controller(
    tokenizer: Tokenizer,
    layer_type: LayerType,
    device: str,
) -> None:
    # load the same model twice so we can verify that each techniques does identical things
    model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", token=True)
    model1 = model1.to(device)  # type: ignore

    model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", token=True)
    model2 = model2.to(device)  # type: ignore

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
    model_patcher.patch_activations(activations, layer_type=layer_type)

    inputs = tokenizer("Hello, world", return_tensors="pt").to(device)
    with torch.no_grad():
        model1_outputs = model1(**inputs, output_hidden_states=False)
        model2_outputs = model2(**inputs, output_hidden_states=False)

    # verify that the outputs are identical
    assert torch.equal(model1_outputs.logits, model2_outputs.logits)


@pytest.mark.parametrize(
    "layer_type",
    [
        "decoder_block",
        "mlp",
        # the other layer types appear to be broken for GPTNeoX in _original_repe
    ],
)
def test_ModelPatcher_patch_activations_piecewise_addition_matches_WrappedReadingVecModel_set_controller(
    tokenizer: Tokenizer,
    layer_type: LayerType,
    device: str,
) -> None:
    # load the same model twice so we can verify that each techniques does identical things
    model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", token=True)
    model1 = model1.to(device)  # type: ignore

    model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", token=True)
    model2 = model2.to(device)  # type: ignore

    assert isinstance(model1, GPTNeoXForCausalLM)  # keep pyright happy
    assert isinstance(model2, GPTNeoXForCausalLM)  # keep pyright happy

    activations = {
        1: torch.randn(1, 512),
        -3: torch.randn(1, 512),
        -4: torch.randn(1, 512),
    }
    layers = list(activations.keys())

    model_patcher = ModelPatcher(model1)
    reading_vec_wrapper = WrappedReadingVecModel(model2, tokenizer)
    reading_vec_wrapper.wrap_block(layers, block_name=layer_type)

    reading_vec_wrapper.set_controller(
        layers, activations, block_name=layer_type, operator="piecewise_linear"
    )
    model_patcher.patch_activations(
        activations, layer_type=layer_type, operator="piecewise_addition"
    )

    inputs = tokenizer("Hello, world", return_tensors="pt").to(device)
    with torch.no_grad():
        model1_outputs = model1(**inputs, output_hidden_states=False)
        model2_outputs = model2(**inputs, output_hidden_states=False)

    # verify that the outputs are identical
    assert torch.equal(model1_outputs.logits, model2_outputs.logits)


@torch.no_grad()
def test_ModelPatcher_patch_activations_with_projection_subtraction(
    model: GPTNeoXForCausalLM,
    tokenizer: Tokenizer,
    device: str,
) -> None:
    # This isn't implemented in the original paper code, despite being in the paper,
    # so we can't test against the original implementation

    inputs = tokenizer("Hello, world", return_tensors="pt").to(device)
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    model_patcher = ModelPatcher(model)
    patch = torch.randn(1, 512).to(device)
    model_patcher.patch_activations({1: patch}, operator="projection_subtraction")
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # The first hidden state is the input embeddings, which are not patched
    assert torch.equal(original_hidden_states[0], patched_hidden_states[0])
    # next is the first decoder block, which is not patched
    assert torch.equal(original_hidden_states[1], patched_hidden_states[1])
    # next is the layer 1, where the patch occurs
    assert not torch.equal(original_hidden_states[2], patched_hidden_states[2])

    projection = (original_hidden_states[2] * patch).sum(-1, keepdim=True) * patch
    patch_norm = patch.norm()
    expected_hidden_state = original_hidden_states[2] - projection / (patch_norm**2)
    assert torch.equal(expected_hidden_state, patched_hidden_states[2])


@torch.no_grad()
def test_ModelPatcher_remove_patches_reverts_model_changes(
    model: GPTNeoXForCausalLM,
    tokenizer: Tokenizer,
    device: str,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt").to(device)
    original_logits = model(**inputs, output_hidden_states=False).logits
    model_patcher = ModelPatcher(model, GptNeoxLayerConfig)
    model_patcher.patch_activations(
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
