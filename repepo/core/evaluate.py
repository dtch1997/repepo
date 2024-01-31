# pyright: strict, reportMissingTypeStubs=false


from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, replace
from statistics import mean
from tqdm import tqdm
from transformers.generation import GenerationConfig
from typing import Callable, Iterable, Optional, Sequence
from repepo.algorithms.repe import SteeringHook
from repepo.core.pipeline import TextProbs
from repepo.core.types import Example
from .pipeline import Pipeline

import numpy as np

# eval_hooks allow us to do custom stuff to the pipeline only during evaluation
EvalHook = Callable[[Pipeline], AbstractContextManager[None]]


def update_completion_template_at_eval(new_template: str) -> EvalHook:
    """Eval hook that changes the completion template for the duration of the evaluation"""

    @contextmanager
    def update_completion_template_hook(pipeline: Pipeline):
        original_template = pipeline.formatter.completion_template
        try:
            pipeline.formatter.completion_template = new_template
            yield
        finally:
            pipeline.formatter.completion_template = original_template

    return update_completion_template_hook


def set_repe_direction_multiplier_at_eval(multiplier: float) -> EvalHook:
    """Eval hook that changes the repetition penalty multiplier for the duration of the evaluation"""

    @contextmanager
    def set_repe_direction_multiplier_hook(pipeline: Pipeline):
        repe_hooks = [hook for hook in pipeline.hooks if isinstance(hook, SteeringHook)]
        if len(repe_hooks) != 1:
            raise ValueError(
                "pipeline must have exactly one SteeringHook to set repe multiplier"
            )
        repe_hook = repe_hooks[0]
        original_multiplier = repe_hook.direction_multiplier
        try:
            repe_hook.direction_multiplier = multiplier
            yield
        finally:
            repe_hook.direction_multiplier = original_multiplier

    return set_repe_direction_multiplier_hook


def select_repe_layer_at_eval(layer: int) -> EvalHook:
    """Eval hook that changes layer to steer for the duration of the evaluation"""

    @contextmanager
    def set_repe_layer_hook(pipeline: Pipeline):
        repe_hooks = [hook for hook in pipeline.hooks if isinstance(hook, SteeringHook)]
        if len(repe_hooks) != 1:
            raise ValueError(
                "pipeline must have exactly one SteeringHook to set repe multiplier"
            )
        steering_vector = repe_hooks[0].steering_vector
        if layer not in steering_vector.layer_activations:
            raise ValueError(f"layer {layer} not found in steering vector")
        original_layer_activations = steering_vector.layer_activations
        try:
            steering_vector.layer_activations = {
                layer: steering_vector.layer_activations[layer]
            }
            yield
        finally:
            steering_vector.layer_activations = original_layer_activations

    return set_repe_layer_hook


@dataclass
class EvalPrediction:
    example: Example
    generated_output: Optional[str] = None
    correct_output_probs: Optional[TextProbs] = None
    incorrect_outputs_probs: Optional[list[TextProbs]] = None

    def get_normalized_correct_probs(self) -> float:
        """
        Normalize the probabilities of correct and incorrect outputs relative to each other
        NOTE: This returns actual probabilities, not logprobs
        NOTE: This assumes that correct_output_probs and incorrect_outputs_probs are not None
        """

        # keep pyright happy
        if self.correct_output_probs is None or self.incorrect_outputs_probs is None:
            raise ValueError("output probs are required to calculate normalized probs")

        # calculate normalized logprobs
        correct_logprob = self.correct_output_probs.sum_logprobs
        incorrect_logprobs = [
            incorrect_probs.sum_logprobs
            for incorrect_probs in self.incorrect_outputs_probs
        ]
        # normalize by max to avoid underflow?
        max_logprob = max([correct_logprob] + incorrect_logprobs)
        correct_logprob = correct_logprob - max_logprob
        incorrect_logprobs = [i - max_logprob for i in incorrect_logprobs]

        # Calculate normalized probability
        correct_prob = np.exp(correct_logprob)
        incorrect_prob = sum([np.exp(i) for i in incorrect_logprobs])
        return correct_prob / (correct_prob + incorrect_prob)


@dataclass
class EvalResult:
    predictions: list[EvalPrediction]
    metrics: dict[str, float]


class Evaluator(ABC):
    requires_generation: bool = False
    requires_probs: bool = False

    @abstractmethod
    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        raise NotImplementedError()


class AccuracyEvaluator(Evaluator):
    """
    Evaluator that computes accuracy, i.e. the percentage of examples where the model
    generated the correct output.
    """

    requires_generation = True

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction, 1 if correct, 0 otherwise."""
        expected = prediction.example.output
        # the output might be longer than the expected depending on how many tokens we generate
        # so just verify that the expected output is a prefix of the generated output
        assert prediction.generated_output is not None, "generation is required"
        is_correct = prediction.generated_output.strip().startswith(expected.strip())
        return 1.0 if is_correct else 0.0

    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        pred_results = [self.score_prediction(pred) for pred in predictions]
        return {"accuracy": mean(pred_results)}


class MultipleChoiceAccuracyEvaluator(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the probability of
    the correct output and comparing it to the probability of the incorrect outputs.
    """

    requires_probs = True

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction, 1 if correct, 0 otherwise."""
        if (
            prediction.example.incorrect_outputs is None
            or len(prediction.example.incorrect_outputs) == 0
        ):
            raise ValueError(
                "Multiple choice evaluation requires examples to set incorrect_outputs"
            )
        # the output might be longer than the expected depending on how many tokens we generate
        # so just verify that the expected output is a prefix of the generated output
        assert prediction.correct_output_probs is not None, "output probs are required"
        assert (
            prediction.incorrect_outputs_probs is not None
        ), "output probs are required"
        correct_prob = prediction.correct_output_probs.sum_logprobs
        incorrect_probs = [
            incorrect_output_probs.sum_logprobs
            for incorrect_output_probs in prediction.incorrect_outputs_probs
        ]
        return 1.0 if correct_prob > max(incorrect_probs) else 0.0

    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        pred_results = [self.score_prediction(pred) for pred in predictions]
        return {"accuracy": mean(pred_results)}


class NormalizedCorrectProbabilityEvaluator(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the average
    normalized probability of the correct output
    """

    requires_probs = True

    def score_prediction(self, prediction: EvalPrediction) -> float:
        return prediction.get_normalized_correct_probs()

    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        pred_results = [self.score_prediction(pred) for pred in predictions]
        return {"average_key_prob": mean(pred_results)}


def evaluate(
    pipeline: Pipeline,
    dataset: Iterable[Example],
    evaluators: Sequence[Evaluator],
    generation_config: Optional[GenerationConfig] = None,
    # these eval_hooks allow us to do custom stuff to the pipeline only during evaluation,
    # e.g. mess with the formatter to use CAA's special answer format
    eval_hooks: Sequence[EvalHook] = [],
    show_progress: bool = True,
    tqdm_desc: str = "Evaluating",
) -> EvalResult:
    # evaluate
    predictions: list[EvalPrediction] = []
    requires_generation = any([e.requires_generation for e in evaluators])
    requires_probs = any([e.requires_probs for e in evaluators])
    with ExitStack() as stack:
        for eval_hook in eval_hooks:
            stack.enter_context(eval_hook(pipeline))
        # TODO: support batching
        for example in tqdm(dataset, disable=not show_progress, desc=tqdm_desc):
            generated_output = None
            correct_output_probs = None
            incorrect_outputs_probs = None
            if requires_generation:
                generated_output = pipeline.generate(
                    example, generation_config=generation_config
                )
            if requires_probs:
                correct_output_probs = pipeline.calculate_output_logprobs(example)
                if example.incorrect_outputs is not None:
                    incorrect_outputs_probs = [
                        pipeline.calculate_output_logprobs(
                            replace(example, output=incorrect_output)
                        )
                        for incorrect_output in example.incorrect_outputs
                    ]
            predictions.append(
                EvalPrediction(
                    example=example,
                    generated_output=generated_output,
                    correct_output_probs=correct_output_probs,
                    incorrect_outputs_probs=incorrect_outputs_probs,
                )
            )
        metrics: dict[str, float] = {}
        for evaluator in evaluators:
            metrics.update(evaluator(predictions))
        return EvalResult(predictions, metrics)
    raise RuntimeError("Should never get here")
