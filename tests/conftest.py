from typing import Any
import pytest
import torch

from transformers import GPTNeoXForCausalLM, AutoTokenizer

from repepo.core.types import Tokenizer, RepExample
from repepo.repe.repe_dataset import bias_dataset


_device: Any = "cuda" if torch.cuda.is_available() else "cpu"


_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
_larger_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

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
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        torch_dtype=torch.float64,
        token=True,
    )
    assert type(model) == GPTNeoXForCausalLM
    return model.to(_device).eval()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return _tokenizer


@pytest.fixture
def repe_toy_dataset() -> list[RepExample]:
    return _repe_toy_dataset


@pytest.fixture
def repe_bias_dataset() -> dict[str, Any]:
    default_user_tag = "[INST]"
    assistant_tag = "[/INST]"
    return bias_dataset(user_tag=default_user_tag, assistant_tag=assistant_tag)


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
    return _larger_tokenizer
