from typing import Any
import pytest
import torch

from transformers import GPTNeoXForCausalLM, AutoTokenizer

from repepo.core.types import Tokenizer, RepExample
from repepo.repe.repe_dataset import bias_dataset


_model: GPTNeoXForCausalLM | None = None
_larger_model: GPTNeoXForCausalLM | None = None


def _load_model() -> GPTNeoXForCausalLM:
    global _model

    if _model is None:
        _pretrained_model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m",
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
            token=True,
        )
        assert type(_pretrained_model) == GPTNeoXForCausalLM
        device: Any = "cuda" if torch.cuda.is_available() else "cpu"
        _larger_model = _pretrained_model.to(device).eval()
    return _larger_model


_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
_larger_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
# _model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", torch_dtype=torch.bfloat16, device_map="auto", token=True, cache_dir = "/ext_usb").eval()
# _tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b", cache_dir = "/ext_usb")

_repe_toy_dataset = [
    RepExample(
        instruction="",
        input=" Consider the bias of the following scenario.\nScenario: All people from eriteria are hardworking\nThe amount of bias in the scenario is  ",
        output="",
        id=0,
        direction=True,
    ),
    RepExample(
        instruction="",
        input=" Consider the bias of the following scenario.\nScenario: All people from eriteria are lazy\nThe amount of bias in the scenario is  ",
        output="",
        id=0,
        direction=False,
    ),
    RepExample(
        instruction="",
        input=" Consider the bias of the following scenario.\nScenario: Ghanaian economy has more to do with the world economy than we think.\nThe amount of bias in the scenario is  ",
        output="",
        id=1,
        direction=True,
    ),
    RepExample(
        instruction="",
        input=" Consider the bias of the following scenario.\nScenario: Ghanaian economy has less to do with the world economy than we think.\nThe amount of bias in the scenario is  ",
        output="",
        id=1,
        direction=False,
    ),
]


@pytest.fixture
def model() -> GPTNeoXForCausalLM:
    return _load_model()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return _tokenizer


@pytest.fixture
def repe_toy_dataset():
    return _repe_toy_dataset


@pytest.fixture
def repe_bias_dataset():
    default_user_tag = "[INST]"
    assistant_tag = "[/INST]"
    return bias_dataset(user_tag=default_user_tag, assistant_tag=assistant_tag)


@pytest.fixture
def larger_model() -> GPTNeoXForCausalLM:
    return _load_larger_model()


@pytest.fixture
def larger_tokenizer() -> Tokenizer:
    return _larger_tokenizer
