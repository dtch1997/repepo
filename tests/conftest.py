from typing import Any
import pytest
import torch

from transformers import GPTNeoXForCausalLM, AutoTokenizer

from repepo.core.types import Tokenizer


_device: Any = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model() -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        torch_dtype=torch.float64,
        token=True,
    )
    assert type(model) == GPTNeoXForCausalLM
    return model.to(_device).eval()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")


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
