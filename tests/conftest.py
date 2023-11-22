from tokenizers import Tokenizer
import pytest
from torch import nn
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


_model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m",
    # torch_dtype=torch.bfloat16,
    torch_dtype=torch.float64,
    device_map="auto",
    token=True,
).eval()
_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
# _model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", torch_dtype=torch.bfloat16, device_map="auto", token=True, cache_dir = "/ext_usb").eval()
# _tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b", cache_dir = "/ext_usb")


@pytest.fixture
def model() -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _model.to(device)


@pytest.fixture
def tokenizer() -> Tokenizer:
    return _tokenizer
