from steering_vectors import SteeringVector
import torch
import math
from transformers import GPTNeoXForCausalLM

from repepo.core.types import Example, Completion, Tokenizer
from repepo.core.pipeline import Pipeline, TextProbs, TokenProb
from repepo.core.format import IdentityFormatter
from repepo.core.evaluate import (
    select_repe_layer_at_eval,
    update_completion_template_at_eval,
    set_repe_direction_multiplier_at_eval,
    EvalPrediction,
    MultipleChoiceAccuracyEvaluator,
    NormalizedPositiveProbabilityEvaluator,
    LogitDifferenceEvaluator,
)
from repepo.core.hook import SteeringHook


def test_update_completion_template_at_eval_hook(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    formatter = IdentityFormatter()
    pipeline = Pipeline(model, tokenizer, formatter=formatter)
    completion = Completion(prompt="hello", response="world")

    pre_hook_output = pipeline.build_full_prompt(completion)
    assert pre_hook_output == "hello world"

    hook = update_completion_template_at_eval(
        "Input: {prompt}\n Output: My answer is: {response}"
    )
    with hook(pipeline):
        new_output = pipeline.build_full_prompt(completion)
        assert new_output == "Input: hello\n Output: My answer is: world"

    # the pipeline should be restored to its original state after the hook exits
    post_hook_output = pipeline.build_full_prompt(completion)
    assert post_hook_output == "hello world"


def test_set_repe_direction_multiplier_at_eval_hook(
    model: GPTNeoXForCausalLM, tokenizer: Tokenizer
) -> None:
    pipeline = Pipeline(model, tokenizer)
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
    pipeline = Pipeline(model, tokenizer)
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


def test_MultipleChoiceAccuracyEvaluator_score_prediction() -> None:
    positive_completion = Completion(prompt="hello", response="world")
    negative_completion = Completion(prompt="hello", response="dear")
    positive_text_probs = TextProbs(
        text="hello world",
        token_probs=[
            TokenProb(token_id=1234, text="hello", logprob=-10, logit=-1),
            TokenProb(token_id=1235, text="world", logprob=-20, logit=-2),
        ],
    )
    negative_text_probs = TextProbs(
        text="hello dear",
        token_probs=[
            TokenProb(token_id=1234, text="hello", logprob=-10, logit=-1),
            TokenProb(token_id=1235, text="dear", logprob=-50, logit=-5),
        ],
    )
    example = Example(positive=positive_completion, negative=negative_completion)

    eval_prediction = EvalPrediction(
        positive_output_prob=positive_text_probs,
        negative_output_prob=negative_text_probs,
        metrics={},
    )
    evaluator = MultipleChoiceAccuracyEvaluator()
    eval_result = evaluator.score_prediction(eval_prediction)
    assert eval_result == 1


def test_LogitDifferenceEvaluator_score_prediction() -> None:
    positive_completion = Completion(prompt="hello", response="world")
    negative_completion = Completion(prompt="hello", response="dear")
    positive_text_probs = TextProbs(
        text="hello world",
        token_probs=[
            TokenProb(token_id=1234, text="hello", logprob=-10, logit=-1),
            TokenProb(token_id=1235, text="world", logprob=-20, logit=-2),
        ],
    )
    negative_text_probs = TextProbs(
        text="hello dear",
        token_probs=[
            TokenProb(token_id=1234, text="hello", logprob=-10, logit=-1),
            TokenProb(token_id=1235, text="dear", logprob=-50, logit=-5),
        ],
    )
    example = Example(positive=positive_completion, negative=negative_completion)

    eval_prediction = EvalPrediction(
        positive_output_prob=positive_text_probs,
        negative_output_prob=negative_text_probs,
        metrics={},
    )
    evaluator = LogitDifferenceEvaluator()
    eval_result = evaluator.score_prediction(eval_prediction)
    assert eval_result == (-1 - 2) - (-1 - 5)


def test_NormalizedPositiveProbabilityEvaluator_score_prediction() -> None:
    positive_completion = Completion(prompt="hello", response="world")
    negative_completion = Completion(prompt="hello", response="dear")
    positive_text_probs = TextProbs(
        text="hello world",
        token_probs=[
            TokenProb(token_id=1234, text="hello", logprob=-10, logit=-1),
            TokenProb(token_id=1235, text="world", logprob=-20, logit=-2),
        ],
    )
    negative_text_probs = TextProbs(
        text="hello dear",
        token_probs=[
            TokenProb(token_id=1234, text="hello", logprob=-10, logit=-1),
            TokenProb(token_id=1236, text="dear", logprob=-50, logit=-5),
        ],
    )
    example = Example(positive=positive_completion, negative=negative_completion)

    eval_prediction = EvalPrediction(
        positive_output_prob=positive_text_probs,
        negative_output_prob=negative_text_probs,
        metrics={},
    )
    evaluator = NormalizedPositiveProbabilityEvaluator()
    eval_result = evaluator.score_prediction(eval_prediction)
    expected_score = math.exp(-10 - 20) / (math.exp(-10 - 20) + math.exp(-10 - 50))
    assert math.isclose(eval_result, expected_score, rel_tol=1e-5)
