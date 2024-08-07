import pytest
from pytest_mock import MockerFixture
from typing import Literal
import torch

from transformers import (
    GPTNeoXForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    LlamaConfig,
)
from repepo.core.types import Tokenizer
from repepo.data.multiple_choice.make_caa_truthfulqa import make_truthfulqa_caa
from repepo.data.multiple_choice.make_caa_sycophancy import make_sycophancy_caa
from repepo.data.multiple_choice.make_mwe_xrisk import make_mwe as make_mwe_xrisk_caa
from repepo.data.multiple_choice.make_mwe_persona import make_mwe_personas_caa


_device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"


# create persona datasets dynamically for testing
@pytest.fixture(scope="package")
def datasets() -> None:
    make_sycophancy_caa()
    make_truthfulqa_caa()
    make_mwe_xrisk_caa()
    make_mwe_personas_caa()


# mock openai.OpenAI for testing
@pytest.fixture(autouse=True)
def get_openai_client_mock(mocker: MockerFixture):
    return mocker.patch("repepo.translation.gpt4_translate.get_openai_client")


@pytest.fixture
def device() -> Literal["cpu", "cuda"]:
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
    return LlamaForCausalLM(config).to(_device).eval()


@pytest.fixture
def llama_chat_tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


@pytest.fixture
def qwen_chat_tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")


@pytest.fixture
def llama3_chat_tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
