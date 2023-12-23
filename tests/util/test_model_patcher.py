from typing import cast
import pytest
import torch
from torch import nn
from transformers import GPTNeoXForCausalLM, GPT2LMHeadModel, LlamaForCausalLM
from repepo.core.types import Model, Tokenizer

from repepo.utils.model_patcher import (
    Gpt2LayerConfig,
    GptNeoxLayerConfig,
    LayerType,
    LlamaLayerConfig,
    ModelPatcher,
    check_predefined_layer_configs,
    enhance_model_config_matchers,
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
    model_patcher.patch_activations(activations, layer_type=layer_type)

    inputs = tokenizer("Hello, world", return_tensors="pt")
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

    model_patcher = ModelPatcher(model1)
    reading_vec_wrapper = WrappedReadingVecModel(model2, tokenizer)
    reading_vec_wrapper.wrap_block(layers, block_name=layer_type)

    reading_vec_wrapper.set_controller(
        layers, activations, block_name=layer_type, operator="piecewise_linear"
    )
    model_patcher.patch_activations(
        activations, layer_type=layer_type, operator="piecewise_addition"
    )

    inputs = tokenizer("Hello, world", return_tensors="pt")
    with torch.no_grad():
        model1_outputs = model1(**inputs, output_hidden_states=False)
        model2_outputs = model2(**inputs, output_hidden_states=False)

    # verify that the outputs are identical
    assert torch.equal(model1_outputs.logits, model2_outputs.logits)


@torch.no_grad()
def test_ModelPatcher_patch_activations_with_projection_subtraction(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    # This isn't implemented in the original paper code, despite being in the paper,
    # so we can't test against the original implementation

    inputs = tokenizer("Hello, world", return_tensors="pt")
    inputs = inputs.to(model.device)
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    model_patcher = ModelPatcher(model)
    patch = torch.randn(1, 512).to(model.device)
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
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt")
    inputs = inputs.to(model.device)
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


def test_check_predefined_layer_configs_matches_gpt2(
    gpt2_model: GPT2LMHeadModel,
) -> None:
    assert check_predefined_layer_configs(gpt2_model) == Gpt2LayerConfig


def test_check_predefined_layer_configs_matches_pythia(
    model: GPTNeoXForCausalLM,
) -> None:
    assert check_predefined_layer_configs(model) == GptNeoxLayerConfig


def test_check_predefined_layer_configs_matches_llama(
    empty_llama_model: LlamaForCausalLM,
) -> None:
    assert check_predefined_layer_configs(empty_llama_model) == LlamaLayerConfig


def test_check_predefined_layer_configs_returns_None_on_no_match() -> None:
    class UnknownModel(nn.Module):
        pass

    unknown_model = cast(Model, UnknownModel())
    assert check_predefined_layer_configs(unknown_model) is None


def test_enhance_model_config_matchers_guesses_fields_if_not_provided(
    model: GPTNeoXForCausalLM,
) -> None:
    enhanced_config = enhance_model_config_matchers(model, {})
    # it should correctly guess every field, resulting in the correct GptNeoxLayerConfig
    assert enhanced_config == GptNeoxLayerConfig


def test_enhance_model_config_matchers_leaves_provided_fields_as_is(
    model: GPTNeoXForCausalLM,
) -> None:
    enhanced_config = enhance_model_config_matchers(
        model, {"decoder_block": "my.{num}.matcher"}
    )
    assert enhanced_config["decoder_block"] == "my.{num}.matcher"
