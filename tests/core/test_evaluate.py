from steering_vectors import SteeringVector
import torch
from transformers import GPTNeoXForCausalLM

from repepo.core.types import Example, Tokenizer
from repepo.core.pipeline import Pipeline, TextProbs, TokenProb
from repepo.core.format import InputOutputFormatter
from repepo.core.evaluate import (
    EvalPrediction,
    select_repe_layer_at_eval,
    update_completion_template_at_eval,
    set_repe_direction_multiplier_at_eval,
)
from repepo.algorithms.repe import SteeringHook


def test_update_completion_template_at_eval_hook(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    formatter = InputOutputFormatter(completion_template="{prompt} {response}")
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    example = Example(instruction="", input="hello", output="world")

    pre_hook_output = pipeline.build_full_prompt(example)
    assert pre_hook_output == "Input:  hello \nOutput: world"

    hook = update_completion_template_at_eval("{prompt} My answer is: {response}")
    with hook(pipeline):
        new_output = pipeline.build_full_prompt(example)
        assert new_output == "Input:  hello \nOutput: My answer is: world"

    # the pipeline should be restored to its original state after the hook exits
    post_hook_output = pipeline.build_full_prompt(example)
    assert post_hook_output == "Input:  hello \nOutput: world"


def test_set_repe_direction_multiplier_at_eval_hook(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    formatter = InputOutputFormatter(completion_template="{prompt} {response}")
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    steering_hook = SteeringHook(
        steering_vector=SteeringVector(layer_activations={1: torch.randn(768)}),
        direction_multiplier=1.0,
        patch_generation_tokens_only=False,
        skip_first_n_generation_tokens=0,
        layer_config=None,
    )
    pipeline.hooks.append(steering_hook)

    set_repe_multiplier_hook = set_repe_direction_multiplier_at_eval(0.5)
    with set_repe_multiplier_hook(pipeline):
        # while applying the hook, the steering vector should have a multiplier of 0.5
        assert steering_hook.direction_multiplier == 0.5

    # the pipeline should be restored to its original state after the hook exits
    assert steering_hook.direction_multiplier == 1.0


def test_select_repe_layer_at_eval_hook(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    formatter = InputOutputFormatter(completion_template="{prompt} {response}")
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    layer_activations = {1: torch.randn(768), 2: torch.randn(768)}
    steering_hook = SteeringHook(
        steering_vector=SteeringVector(layer_activations=layer_activations),
        direction_multiplier=1.0,
        patch_generation_tokens_only=False,
        skip_first_n_generation_tokens=0,
        layer_config=None,
    )
    pipeline.hooks.append(steering_hook)

    set_repe_multiplier_hook = select_repe_layer_at_eval(1)
    with set_repe_multiplier_hook(pipeline):
        # while applying the hook, the steering vector only have the layer 1 activations
        assert steering_hook.steering_vector.layer_activations == {
            1: layer_activations[1]
        }

    # the pipeline should be restored to its original state after the hook exits
    assert steering_hook.steering_vector.layer_activations == layer_activations


def test_EvalPrediction_get_normalized_correct_probs_equally_splits_equal_probs() -> (
    None
):
    example = Example(instruction="", input="hello", output="world")
    text_probs = TextProbs(
        text="hello world",
        token_probs=[
            TokenProb(token_id=1234, text="hello", logprob=-20),
            TokenProb(token_id=1235, text="world", logprob=-11),
        ],
    )
    prediction = EvalPrediction(
        example=example,
        correct_output_probs=text_probs,
        incorrect_outputs_probs=[
            text_probs,
            text_probs,
            text_probs,
        ],
    )

    # since both correct and 3 incorrect have identical probs, the normalized correct probs should be 0.25 (1 / (1 + 3))
    assert prediction.get_normalized_correct_probs() == 0.25
