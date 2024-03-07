import logging

from repepo.steering.utils.helpers import SteeringResult
from repepo.core.types import Dataset
from repepo.core.pipeline import Pipeline
from repepo.core.evaluate import (
    update_completion_template_at_eval,
    select_repe_layer_at_eval,
    set_repe_direction_multiplier_at_eval,
    evaluate,
    MultipleChoiceAccuracyEvaluator,
    LogitDifferenceEvaluator,
    NormalizedPositiveProbabilityEvaluator,
)
from repepo.core.hook import SteeringHook
from steering_vectors import SteeringVector, guess_and_enhance_layer_config


def evaluate_steering_vector(
    pipeline: Pipeline,
    steering_vector: SteeringVector,
    dataset: Dataset,
    layers: list[int],
    multipliers: list[float],
    patch_generation_tokens_only: bool = True,
    skip_first_n_generation_tokens: int = 0,
    completion_template: str | None = None,
    logger: logging.Logger | None = None,
) -> list[SteeringResult]:
    results = []

    # Create steering hook and add it to pipeline
    steering_hook = SteeringHook(
        steering_vector=steering_vector,
        direction_multiplier=0,
        patch_generation_tokens_only=patch_generation_tokens_only,
        skip_first_n_generation_tokens=skip_first_n_generation_tokens,
        layer_config=guess_and_enhance_layer_config(pipeline.model),
    )
    pipeline.hooks.append(steering_hook)

    for layer_id in layers:
        for multiplier in multipliers:
            eval_hooks = [
                set_repe_direction_multiplier_at_eval(multiplier),
                select_repe_layer_at_eval(layer_id),
            ]
            if completion_template is not None:
                eval_hooks.append(
                    update_completion_template_at_eval(completion_template)
                )

            # Run evaluate to get metrics
            result = evaluate(
                pipeline,
                dataset,
                eval_hooks=eval_hooks,
                evaluators=[
                    MultipleChoiceAccuracyEvaluator(),
                    LogitDifferenceEvaluator(),
                    NormalizedPositiveProbabilityEvaluator(),
                ],
                logger=logger,
            )

            result = SteeringResult(
                layer_id=layer_id,
                multiplier=multiplier,
                mcq_acc=result.metrics["mcq_acc"],
                logit_diff=result.metrics["logit_diff"],
                pos_prob=result.metrics["pos_prob"],
            )
            results.append(result)
            if logger is not None:
                logger.info(
                    f"Layer {layer_id}, multiplier {multiplier:.2f}: "
                    f"MCQ Accuracy {result.mcq_acc:.2f} "
                    f"Positive Prob {result.pos_prob:.2f} "
                    f"Logit Diff {result.logit_diff:.2f} "
                )

    return results
