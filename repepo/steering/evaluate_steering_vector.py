import logging


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
    EvalResult,
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
    evaluators: list = [
        MultipleChoiceAccuracyEvaluator(),
        LogitDifferenceEvaluator(),
        NormalizedPositiveProbabilityEvaluator(),
    ],
    show_progress: bool = True,
) -> list[EvalResult]:
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
                evaluators=evaluators,
                logger=logger,
                show_progress=show_progress,
            )
            results.append(result)
            if logger is not None:
                metrics_info_str = ""
                for metric, value in result.metrics.items():
                    metrics_info_str += f"{metric} {value:.2f} "
                logger.info(
                    f"Layer {layer_id}, multiplier {multiplier:.2f}: "
                    + metrics_info_str
                )

    pipeline.hooks.clear()

    return results
