import torch

from transformers.models.gpt_neox import GPTNeoXForCausalLM
from repepo.core.types import Tokenizer
from steering_vectors.layer_matching import GptNeoxLayerConfig

from steering_vectors.steering_vector import SteeringVector


@torch.no_grad()
def test_SteeringVector_patch_activations(
    model: GPTNeoXForCausalLM,
    tokenizer: Tokenizer,
    device: str,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt").to(device)
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    patch = torch.randn(512).to(device)
    steering_vector = SteeringVector(
        layer_activations={1: patch},
        layer_type="decoder_block",
    )
    steering_vector.patch_activations(model)
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # The first hidden state is the input embeddings, which are not patched
    assert torch.equal(original_hidden_states[0], patched_hidden_states[0])
    # next is the first decoder block, which is not patched
    assert torch.equal(original_hidden_states[1], patched_hidden_states[1])
    # next is the layer 1, where the patch occurs
    assert not torch.equal(original_hidden_states[2], patched_hidden_states[2])

    expected_hidden_state = original_hidden_states[2] + patch
    assert torch.equal(expected_hidden_state, patched_hidden_states[2])


@torch.no_grad()
def test_SteeringVector_patch_activations_with_min_token_index(
    model: GPTNeoXForCausalLM,
    tokenizer: Tokenizer,
    device: str,
) -> None:
    inputs = tokenizer(
        "What is cheesier than cheese? Nothing is cheesier than cheese",
        return_tensors="pt",
    ).to(device)
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    patch = torch.randn(512).to(device)
    steering_vector = SteeringVector(
        layer_activations={1: patch},
        layer_type="decoder_block",
    )
    steering_vector.patch_activations(model, min_token_index=5)
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # The first 5 tokens should not be patched, due to min_token_index
    assert torch.equal(
        original_hidden_states[2][0, :5], patched_hidden_states[2][0, :5]
    )
    assert not torch.equal(
        original_hidden_states[2][0, 5:], patched_hidden_states[2][0, 5:]
    )

    expected_hidden_state = original_hidden_states[2][0, 5:] + patch
    assert torch.equal(expected_hidden_state, patched_hidden_states[2][0, 5:])


@torch.no_grad()
def test_SteeringVector_handle_reverts_model_changes(
    model: GPTNeoXForCausalLM,
    tokenizer: Tokenizer,
    device: str,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt").to(device)
    original_logits = model(**inputs, output_hidden_states=False).logits
    steering_vector = SteeringVector(
        layer_activations={
            1: torch.randn(512),
            -1: torch.randn(512),
        },
        layer_type="decoder_block",
    )
    handle = steering_vector.patch_activations(model, layer_config=GptNeoxLayerConfig)
    patched_logits = model(**inputs, output_hidden_states=False).logits
    handle.remove()
    unpatched_logits = model(**inputs, output_hidden_states=False).logits

    assert not torch.equal(original_logits, patched_logits)
    assert torch.equal(original_logits, unpatched_logits)
