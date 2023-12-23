import torch
from transformers.models.gpt_neox import GPTNeoXForCausalLM
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from transformers.pipelines import TextGenerationPipeline

from repepo.core.types import Model, Tokenizer
from repepo.repe.rep_control_pipeline import RepControlPipeline

from tests._original_repe.rep_control_pipeline import (
    RepControlPipeline as OriginalRepControlPipeline,
)


def test_our_hacked_RepControlPipeline_matches_original_behavior(
    tokenizer: Tokenizer,
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

    our_pipeline = RepControlPipeline(
        model=model1,
        tokenizer=tokenizer,
        layers=layers,
        block_name="decoder_block",
    )
    original_pipeline = OriginalRepControlPipeline(
        model=model2,
        tokenizer=tokenizer,
        layers=layers,
        block_name="decoder_block",
    )

    inputs = "12345"
    our_outputs = our_pipeline(
        inputs,
        activations=activations,
        batch_size=4,
        max_new_tokens=5,
        do_sample=False,
    )
    original_outputs = original_pipeline(
        inputs,
        activations=activations,
        batch_size=4,
        max_new_tokens=5,
        do_sample=False,
    )
    assert our_outputs == original_outputs


def test_RepControlPipeline_matches_base_huggingface_outputs_if_no_activations_are_patched(
    model: Model, tokenizer: GPTNeoXTokenizerFast
) -> None:
    inputs = "12345"
    hf_pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    hf_outputs = hf_pipeline(
        inputs,
        batch_size=4,
        max_new_tokens=5,
        do_sample=False,
    )

    our_pipeline = RepControlPipeline(
        model=model,
        tokenizer=tokenizer,
        layers=[],
        block_name="decoder_block",
    )
    our_outputs = our_pipeline(
        inputs,
        activations=None,
        batch_size=4,
        max_new_tokens=5,
        do_sample=False,
    )
    assert our_outputs == hf_outputs
