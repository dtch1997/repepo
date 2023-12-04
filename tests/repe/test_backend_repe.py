from tokenizers import Tokenizer
from transformers import GPTNeoXForCausalLM
from repepo.repe.repe_dataset import bias_dataset
from repepo.repe.rep_control_pipeline import RepControlPipeline
from repepo.repe.rep_reading_pipeline import RepReadingPipeline
import torch
import math
import pytest


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Cuda not available"
            ),
        ),
        pytest.param(
            "cpu",
            marks=pytest.mark.skipif(
                torch.cuda.is_available(), reason="Cuda is available"
            ),
        ),
    ],
)
def test_rep_readers_and_control(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer, device: str
) -> None:
    """
    Test that the rep readers work for Pythia 70m with double precision
    """
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = "pca"
    block_name = "decoder_block"
    control_method = "reading_vec"

    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.bos_token_id = 1

    rep_reading_pipeline = RepReadingPipeline(
        model=model, tokenizer=tokenizer, device=device
    )

    default_user_tag = "[INST]"
    assistant_tag = "[/INST]"
    dataset = bias_dataset(user_tag=default_user_tag, assistant_tag=assistant_tag)

    train_data = dataset["train"]
    rep_reader = rep_reading_pipeline.get_directions(
        train_data["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=n_difference,
        train_labels=train_data["labels"],
        direction_method=direction_method,
        batch_size=2
    )

    assert rep_reader is not None
    assert rep_reader.directions is not None
    assert math.isclose(rep_reader.directions[-3][0][0], 0.00074, abs_tol=1e-5)

    rep_control_pipeline = RepControlPipeline(
        model=model,
        tokenizer=tokenizer,
        layers=hidden_layers,
        block_name=block_name,
        control_method=control_method,
        device=device,
    )

    coeff = 0.1
    max_new_tokens = 64

    activations = {}
    for layer in hidden_layers:
        activations[layer] = (
            torch.tensor(
                coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
            ).to(model.device)
            # .bfloat16()
        )

    assert activations[-3].shape == torch.Size([1, 512])
    assert math.isclose(
        float(activations[-3][0][0]), 7.410049147438258e-05, abs_tol=1e-6
    )

    inputs = "12345"
    control_outputs = rep_control_pipeline(
        inputs,
        activations=activations,
        batch_size=4,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    assert (
        control_outputs[0]["generated_text"]
        == "123456789_1\n\n#define S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S"
    )

    # TODO: remove this
    from functools import partial
    model_wrapper = partial(
        rep_control_pipeline,
        activations=activations,
        batch_size=4,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    breakpoint()


