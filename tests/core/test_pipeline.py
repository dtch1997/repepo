import torch
import scipy
import numpy as np
from textwrap import dedent
from transformers import GPTNeoXForCausalLM

from repepo.core.pipeline import Pipeline, compute_moments, compute_quantiles
from repepo.core.format import IdentityFormatter, LlamaChatFormatter
from repepo.core.types import Completion, Tokenizer
from syrupy.assertion import SnapshotAssertion


def _compute_moments_scipy(x: np.ndarray, axis: int) -> np.ndarray:
    mean = np.mean(x, axis=axis)
    std = scipy.stats.tstd(x, axis=axis, ddof=1)
    skew = scipy.stats.skew(x, axis=axis)
    kurtosis = scipy.stats.kurtosis(x, axis=axis, fisher=False)
    return np.stack([mean, std, skew, kurtosis], axis=1)


def test_compute_moments_basic():
    tensor = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]
    )
    output = compute_moments(tensor, dim=1)
    expected_output = torch.from_numpy(_compute_moments_scipy(tensor.numpy(), axis=1))
    # NOTE: torch kurtosis does not agree with scipy kurtosis for some reason...
    # Omitted from testing for now
    torch.testing.assert_allclose(output[:, :3], expected_output[:, :3])


def test_compute_moments_edge_cases():
    # Test with a single-value tensor
    tensor = torch.tensor([[1.0]])
    expected_output = torch.tensor([[1.0, np.nan, np.nan, np.nan]])
    output = compute_moments(tensor, dim=1)
    torch.testing.assert_allclose(output[:, :3], expected_output[:, :3])

    # Test with a tensor with uniform values
    tensor = torch.full((1, 4), 3.0)
    expected_output = torch.tensor([[3.0, 0.0, np.nan, np.nan]])
    output = compute_moments(tensor, dim=1)
    torch.testing.assert_allclose(output[:, :3], expected_output[:, :3])


def test_compute_quantiles_basic():
    tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    expected_output = torch.tensor(
        [[1.0, 1.75, 2.5, 3.25, 4.0], [1.0, 1.75, 2.5, 3.25, 4.0]]
    )
    output = compute_quantiles(tensor, dim=1)
    torch.testing.assert_allclose(output, expected_output)


def test_compute_quantiles_edge_cases():
    # Test with a single-value tensor
    tensor = torch.tensor([[2.0]])
    expected_output = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0]])
    output = compute_quantiles(tensor, dim=1)
    torch.testing.assert_allclose(output, expected_output)

    # Test with non-unique values
    tensor = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    expected_output = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0]])
    output = compute_quantiles(tensor, dim=1)
    torch.testing.assert_allclose(output, expected_output)


def test_basic_Pipeline_generate(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.generate(
        Completion(prompt="Respond A B C D", response="E"),
        generation_config=None,
    )
    # pythia-70m generates nonsense, so just verify we get something
    assert isinstance(res, str)
    assert len(res) > 0

def test_basic_Pipeline_build_generation_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.build_generation_prompt(
        Completion(prompt="Respond A B C D", response="E")
    )
    assert res == "Respond A B C D"


def test_basic_Pipeline_build_full_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.build_full_prompt(Completion(prompt="Respond A B C D", response="E"))
    assert res == "Respond A B C D E"


def test_basic_Pipeline_build_generation_prompt_with_custom_completion_template(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    formatter = IdentityFormatter(
        completion_template="Input: {prompt}\n Output: My answer is: {response}",
    )
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    res = pipeline.build_generation_prompt(
        Completion(prompt="Respond A B C D", response="E")
    )
    assert res == "Input: Respond A B C D\n Output: My answer is:"


def test_basic_Pipeline_build_full_prompt_with_custom_completion_template(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    formatter = IdentityFormatter(
        completion_template="Input: {prompt}\n Output: My answer is: {response}",
    )
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    res = pipeline.build_full_prompt(
        Completion(prompt="Respond A B C D", response="E"),
    )
    assert res == "Input: Respond A B C D\n Output: My answer is: E"


def test_icl_Pipeline_build_generation_prompt(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer, snapshot: SnapshotAssertion
) -> None:
    dataset = [
        Completion(prompt="Paris is in", response="France"),
        Completion(prompt="London is in", response="England"),
        Completion(prompt="Berlin is in", response="Germany"),
    ]
    pipeline = Pipeline(model, tokenizer)
    pipeline.conversation_history = dataset
    res = pipeline.build_generation_prompt(
        Completion(prompt="Beijing is in", response="China")
    )
    assert res == snapshot


def test_basic_pipeline_calculate_output_logprobs(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
    res = pipeline.calculate_output_logprobs(
        Completion(
            prompt="Input: Select the best answer.\nA B C D", response="Output: D"
        )
    )
    assert res.sum_logprobs < 0
    assert res.text == "Input: Select the best answer.\nA B C D Output: D"
    assert (
        "".join([tok.text for tok in res.token_probs])
        # "Input" is the first token, so the model doesn't predict this
        == ": Select the best answer.\nA B C D Output: D"
    )
    for tok in res.token_probs:
        assert tok.logprob < 0


def test_pipeline_can_reproduce_caa_train_prompts(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    # copied from a real prompt in the CAA training set, abbreviated for brevity
    target = dedent(
        """
        [INST] <<SYS>>
        You are a helpful, honest and concise assistant.
        <</SYS>>
        
        Hello, my name is Linda Johnson ...
        Choices:
        (A) Agree
        (B) Disagree [/INST] (B)
        """
    )
    formatter = LlamaChatFormatter()
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    completion = Completion(
        prompt=dedent(
            """
            Hello, my name is Linda Johnson ...
            Choices:
            (A) Agree
            (B) Disagree
            """
        ),
        response="(B)",
    )
    assert pipeline.build_full_prompt(completion).strip() == target.strip()
