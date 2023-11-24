from typing import cast
from transformers import PreTrainedTokenizer
import pytest
import torch

from transformers import GPTNeoXForCausalLM, AutoTokenizer


_model: GPTNeoXForCausalLM | None = None


def _load_model() -> GPTNeoXForCausalLM:
    global _model

    if _model is None:
        _pretrained_model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m",
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float64,
            device_map="auto",
            token=True,
        )
        assert type(_pretrained_model) == GPTNeoXForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _pretrained_model.eval()
    return _model


_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
# _model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", torch_dtype=torch.bfloat16, device_map="auto", token=True, cache_dir = "/ext_usb").eval()
# _tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b", cache_dir = "/ext_usb")


@pytest.fixture
def model() -> GPTNeoXForCausalLM:
    return _load_model()


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    # TODO: figure out what the corrrect type to use is here
    return cast(PreTrainedTokenizer, _tokenizer)
