from transformers import GPTNeoXForCausalLM, LlamaForCausalLM, GPT2LMHeadModel

from repepo.utils.layer_matching import (
    _guess_block_matcher_from_layers,
    guess_decoder_block_matcher,
    guess_input_layernorm_matcher,
    guess_mlp_matcher,
    guess_self_attn_matcher,
    guess_post_attention_layernorm_matcher,
)


def test_guess_block_matcher_from_layers() -> None:
    layers = [
        "x.e",
        "x.y.0",
        "x.y.0.attn",
        "x.y.1",
        "x.y.1.attn",
        "x.y.2",
        "x.y.2.attn",
        "x.lm_head",
    ]
    assert _guess_block_matcher_from_layers(layers) == "x.y.{num}"


def test_guess_matchers_for_llama(
    empty_llama_model: LlamaForCausalLM,
) -> None:
    assert guess_decoder_block_matcher(empty_llama_model) == "model.layers.{num}"
    assert guess_self_attn_matcher(empty_llama_model) == "model.layers.{num}.self_attn"
    assert guess_mlp_matcher(empty_llama_model) == "model.layers.{num}.mlp"
    assert (
        guess_input_layernorm_matcher(empty_llama_model)
        == "model.layers.{num}.input_layernorm"
    )
    assert (
        guess_post_attention_layernorm_matcher(empty_llama_model)
        == "model.layers.{num}.post_attention_layernorm"
    )


def test_matchers_for_pythia(model: GPTNeoXForCausalLM) -> None:
    assert guess_decoder_block_matcher(model) == "gpt_neox.layers.{num}"
    assert guess_self_attn_matcher(model) == "gpt_neox.layers.{num}.attention"
    assert guess_mlp_matcher(model) == "gpt_neox.layers.{num}.mlp"
    assert (
        guess_input_layernorm_matcher(model) == "gpt_neox.layers.{num}.input_layernorm"
    )
    assert (
        guess_post_attention_layernorm_matcher(model)
        == "gpt_neox.layers.{num}.post_attention_layernorm"
    )


def test_guess_matchers_for_gpt2(gpt2_model: GPT2LMHeadModel) -> None:
    assert guess_decoder_block_matcher(gpt2_model) == "transformer.h.{num}"
    assert guess_self_attn_matcher(gpt2_model) == "transformer.h.{num}.attn"
    assert guess_mlp_matcher(gpt2_model) == "transformer.h.{num}.mlp"
    assert guess_input_layernorm_matcher(gpt2_model) == "transformer.h.{num}.ln_1"
    assert (
        guess_post_attention_layernorm_matcher(gpt2_model) == "transformer.h.{num}.ln_2"
    )
