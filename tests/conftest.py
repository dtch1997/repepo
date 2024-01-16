import pytest
import torch

from transformers import (
    GPTNeoXForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    LlamaConfig,
)
from repepo.core.types import Tokenizer


_device: str = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def device() -> str:
    return _device


@pytest.fixture
def model() -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        token=True,
    )
    assert type(model) == GPTNeoXForCausalLM
    return model.to(_device).eval()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        model_max_length=128,  # Required to avoid overflow error in SFT
        padding_side="right",
    )


@pytest.fixture
def larger_model() -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m",
        token=True,
    )
    assert type(model) == GPTNeoXForCausalLM
    return model.to(_device).eval()


@pytest.fixture
def larger_tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")


@pytest.fixture
def gpt2_model() -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    assert type(model) == GPT2LMHeadModel
    return model.to(_device).eval()


@pytest.fixture
def gpt2_tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def empty_llama_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        num_hidden_layers=3,
        hidden_size=1024,
        intermediate_size=2752,
    )
    return LlamaForCausalLM(config)
