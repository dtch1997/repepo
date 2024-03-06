import torch
import simple_parsing


from repepo.experiments.steering.utils.helpers import (
    get_model_name,
    get_model_and_tokenizer,
    get_formatter,
    SteeringConfig,
    SteeringResult,
    load_concept_vectors,
    save_results,
    make_dataset,
    EmptyTorchCUDACache,
)
from repepo.experiments.steering.utils.configs import list_configs

from repepo.core.types import Dataset
from repepo.core.pipeline import Pipeline
from repepo.core.evaluate import (
    print_first_example,
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
    verbose: bool = False,
) -> list[SteeringResult]:
    caa_results = []

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
                    # update_completion_template_at_eval(
                    #     "{prompt} My answer is {response}"
                    # ),
                    print_first_example(),
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

            caa_result = SteeringResult(
                layer_id=layer_id,
                multiplier=multiplier,
                mcq_acc=result.metrics["mcq_acc"],
                logit_diff=result.metrics["logit_diff"],
                pos_prob=result.metrics["pos_prob"],
            )
            caa_results.append(caa_result)
            if verbose:
                print(
                    f"Layer {layer_id}, multiplier {multiplier:.2f}: "
                    f"MCQ Accuracy {caa_result.mcq_acc:.2f} "
                    f"Positive Prob {caa_result.pos_prob:.2f} "
                    f"Logit Diff {caa_result.logit_diff:.2f} "
                )

    return caa_results


def run_load_extract_and_save(
    config: SteeringConfig,
):
    test_dataset = make_dataset(config.test_dataset_name, config.test_split_name)
    model_name = get_model_name(config.use_base_model, config.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)
    formatter = get_formatter(config.formatter)

    if len(config.layers) == 0:
        config.layers = list(range(model.config.num_hidden_layers))
    pipeline = Pipeline(model, tokenizer, formatter)
    concept_vectors = load_concept_vectors(config)

    results = evaluate_steering_with_concept_vectors(
        pipeline=pipeline,
        concept_vectors=concept_vectors,
        dataset=test_dataset,
        layers=config.layers,
        multipliers=config.multipliers,
        verbose=config.verbose,
    )

    save_results(config, results)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(SteeringConfig, dest="config")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args()
    config = args.config

    if args.configs:
        all_configs = list_configs(
            args.configs, config.train_split_name, config.test_split_name
        )
        for config in all_configs:
            # Train and test on the same dataset
            print("Running on dataset: ", config.test_dataset_name)
            with EmptyTorchCUDACache():
                run_load_extract_and_save(config)
    else:
        with EmptyTorchCUDACache():
            run_load_extract_and_save(config)
