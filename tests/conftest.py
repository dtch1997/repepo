from typing import Any
import pytest
import torch

from transformers import GPTNeoXForCausalLM, AutoTokenizer

from repepo.variables import Environ
from repepo.core.types import Tokenizer


_model: GPTNeoXForCausalLM | None = None
_larger_model: GPTNeoXForCausalLM | None = None


def _load_model() -> GPTNeoXForCausalLM:
    global _model

    if _model is None:
        _pretrained_model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m",
            cache_dir=Environ.HuggingfaceCacheDir,
            torch_dtype=torch.float64,
            token=True,
        )
        assert type(_pretrained_model) == GPTNeoXForCausalLM
        device: Any = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _pretrained_model.to(device).eval()
    return _model


def _load_larger_model() -> GPTNeoXForCausalLM:
    global _larger_model

    if _larger_model is None:
        _pretrained_model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-160m",
            cache_dir=Environ.HuggingfaceCacheDir,
            token=True,
        )
        assert type(_pretrained_model) == GPTNeoXForCausalLM
        device: Any = "cuda" if torch.cuda.is_available() else "cpu"
        _larger_model = _pretrained_model.to(device).eval()
    return _larger_model


_tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m",
    cache_dir=Environ.HuggingfaceCacheDir,
    model_max_length=128,
    padding_side="right",
    use_fast=True,
)
_larger_tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-160m",
    cache_dir=Environ.HuggingfaceCacheDir,
    model_max_length=128,
    padding_side="right",
    use_fast=True,
)
# _model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", torch_dtype=torch.bfloat16, device_map="auto", token=True, cache_dir = "/ext_usb").eval()
# _tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b", cache_dir = "/ext_usb")


@pytest.fixture
def model() -> GPTNeoXForCausalLM:
    return _load_model()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return _tokenizer


@pytest.fixture
def larger_model() -> GPTNeoXForCausalLM:
    return _load_larger_model()


@pytest.fixture
def larger_tokenizer() -> Tokenizer:
    return _larger_tokenizer
