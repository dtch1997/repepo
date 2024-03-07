# pyright: strict, reportMissingTypeStubs=false


from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from statistics import mean
from tqdm import tqdm
from typing import Callable, Iterable, Sequence
from repepo.core.hook import SteeringHook
from repepo.core.pipeline import TextProbs
from repepo.core.types import Example
from repepo.core.pipeline import Pipeline

import numpy as np
import logging

# eval_hooks allow us to do custom stuff to the pipeline only during evaluation
EvalHook = Callable[[Pipeline], AbstractContextManager[None]]


def print_first_example() -> EvalHook:
    """Eval hook that prints the first example"""

    @contextmanager
    def print_first_example_hook(pipeline: Pipeline):
        try:
            pipeline.print_first_example = True
            yield
        finally:
            pipeline.print_first_example = False

    return print_first_example_hook


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
    positive_output_prob: TextProbs
    negative_output_prob: TextProbs


@dataclass
class EvalResult:
    predictions: list[EvalPrediction]
    metrics: dict[str, float]


class Evaluator(ABC):
    requires_generation: bool = False
    requires_probs: bool = False

    @abstractmethod
    def get_metric_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score_prediction(self, prediction: EvalPrediction) -> float:
        raise NotImplementedError

    def __call__(self, predictions: Sequence[EvalPrediction]) -> dict[str, float]:
        pred_results = [self.score_prediction(pred) for pred in predictions]
        return {self.get_metric_name(): mean(pred_results)}


class MultipleChoiceAccuracyEvaluator(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the probability of
    the correct output and comparing it to the probability of the incorrect outputs.
    """

    requires_probs = True

    def get_metric_name(self) -> str:
        return "mcq_acc"

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction, 1 if correct, 0 otherwise."""

        # the output might be longer than the expected depending on how many tokens we generate
        # so just verify that the expected output is a prefix of the generated output
        positive_output_prob = prediction.positive_output_prob.sum_logprobs
        negative_output_prob = prediction.negative_output_prob.sum_logprobs
        return 1.0 if positive_output_prob > negative_output_prob else 0.0


class LogitDifferenceEvaluator(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the average difference
    in logit between the correct and incorrect outputs.
    """

    requires_probs = True

    def get_metric_name(self) -> str:
        return "logit_diff"

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """Score a single prediction based on difference in sum of logits."""

        # calculate difference in logits
        positive_output_logit = prediction.positive_output_prob.sum_logits
        negative_output_logit = prediction.negative_output_prob.sum_logits
        return positive_output_logit - negative_output_logit


class NormalizedPositiveProbabilityEvaluator(Evaluator):
    """
    Evaluator that scores multiple choice examples by computing the
    normalized probability of the positive output
    """

    requires_probs = True

    def get_metric_name(self) -> str:
        return "pos_prob"

    def score_prediction(self, prediction: EvalPrediction) -> float:
        """
        Normalize the probabilities of positive and negative outputs relative to each other
        NOTE: This returns actual probabilities, not logprobs
        """

        # calculate normalized logprobs
        positive_output_logprob = prediction.positive_output_prob.sum_logprobs
        negative_output_logprob = prediction.negative_output_prob.sum_logprobs

        # normalize by max to avoid underflow?
        max_logprob = max(positive_output_logprob, negative_output_logprob)
        positive_output_logprob = positive_output_logprob - max_logprob
        negative_output_logprob = negative_output_logprob - max_logprob

        # Calculate normalized probability
        positive_output_prob = np.exp(positive_output_logprob)
        negative_output_prob = np.exp(negative_output_logprob)
        return positive_output_prob / (positive_output_prob + negative_output_prob)


def evaluate(
    pipeline: Pipeline,
    dataset: Iterable[Example],
    evaluators: Sequence[Evaluator],
    # these eval_hooks allow us to do custom stuff to the pipeline only during evaluation,
    # e.g. mess with the formatter to use CAA's special answer format
    eval_hooks: Sequence[EvalHook] = [],
    show_progress: bool = True,
    tqdm_desc: str = "Evaluating",
    logger: logging.Logger | None = None,
) -> EvalResult:
    # evaluate
    predictions: list[EvalPrediction] = []

    with ExitStack() as stack:
        for eval_hook in eval_hooks:
            stack.enter_context(eval_hook(pipeline))
        # TODO: support batching
        for i, example in enumerate(
            tqdm(dataset, disable=not show_progress, desc=tqdm_desc)
        ):
            if logger is not None and i == 0:
                logger.info(
                    f"Example full prompt: \n {pipeline.build_full_prompt(example.positive)}"
                )
            positive_probs = pipeline.calculate_output_logprobs(example.positive)
            negative_probs = pipeline.calculate_output_logprobs(example.negative)

            predictions.append(
                EvalPrediction(
                    example=example,
                    positive_output_prob=positive_probs,
                    negative_output_prob=negative_probs,
                )
            )
        metrics: dict[str, float] = {}
        for evaluator in evaluators:
            metrics.update(evaluator(predictions))
        return EvalResult(predictions, metrics)
    raise RuntimeError("Should never get here")
