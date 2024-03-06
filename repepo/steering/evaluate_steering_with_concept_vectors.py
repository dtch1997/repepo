import torch
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
from steering_vectors import SteeringVector


def evaluate_steering_with_concept_vectors(
    pipeline: Pipeline,
    concept_vectors: dict[int, torch.Tensor],
    dataset: Dataset,
    layers: list[int],
    multipliers: list[float],
    completion_template: str = "{prompt} {response}",
    verbose: bool = False,
) -> list[SteeringResult]:
    results = []
    steering_vector = SteeringVector(layer_activations=concept_vectors)

    # Create steering hook and add it to pipeline
    steering_hook = SteeringHook(
        steering_vector=steering_vector,
        direction_multiplier=0,
        patch_generation_tokens_only=False,
        skip_first_n_generation_tokens=0,
        layer_config=None,
    )
    pipeline.hooks.append(steering_hook)

    for layer_id in layers:
        for multiplier in multipliers:
            pass
            # Run evaluate to get metrics
            result = evaluate(
                pipeline,
                dataset,
                eval_hooks=[
                    # TODO: different datasets need different of evaluating prompt template
                    update_completion_template_at_eval(completion_template),
                    set_repe_direction_multiplier_at_eval(multiplier),
                    select_repe_layer_at_eval(layer_id),
                ],
                evaluators=[
                    MultipleChoiceAccuracyEvaluator(),
                    LogitDifferenceEvaluator(),
                    NormalizedPositiveProbabilityEvaluator(),
                ],
                verbose=True,
            )

            result = SteeringResult(
                layer_id=layer_id,
                multiplier=multiplier,
                mcq_acc=result.metrics["mcq_acc"],
                logit_diff=result.metrics["logit_diff"],
                pos_prob=result.metrics["pos_prob"],
            )
            results.append(result)
            if verbose:
                print(
                    f"Layer {layer_id}, multiplier {multiplier:.2f}: "
                    f"MCQ Accuracy {result.mcq_acc:.2f} "
                    f"Positive Prob {result.pos_prob:.2f} "
                    f"Logit Diff {result.logit_diff:.2f} "
                )

    return results
