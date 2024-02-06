import torch
import simple_parsing

from statistics import mean

from repepo.experiments_2.utils.helpers import (
    get_model_name,
    get_model_and_tokenizer,
    SteeringConfig,
    SteeringResult,
    load_concept_vectors,
    save_results,
    list_subset_of_datasets,
    make_dataset,
    EmptyTorchCUDACache,
)
from repepo.core.format import LlamaChatFormatter
from repepo.core.types import Dataset
from repepo.core.pipeline import Pipeline
from repepo.core.evaluate import (
    select_repe_layer_at_eval,
    set_repe_direction_multiplier_at_eval,
    update_completion_template_at_eval,
    evaluate,
    MultipleChoiceAccuracyEvaluator,
)
from repepo.algorithms.repe import SteeringHook
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
                    update_completion_template_at_eval(
                        "{prompt} My answer is {response}"
                    ),
                    set_repe_direction_multiplier_at_eval(multiplier),
                    select_repe_layer_at_eval(layer_id),
                ],
                evaluators=[MultipleChoiceAccuracyEvaluator()],
            )
            key_probs = [
                pred.get_normalized_correct_probs() for pred in result.predictions
            ]

            caa_result = SteeringResult(
                layer_id=layer_id,
                multiplier=multiplier,
                mcq_accuracy=result.metrics["accuracy"],
                average_key_prob=mean(key_probs),
            )
            caa_results.append(caa_result)
            if verbose:
                print(
                    f"Layer {layer_id}, multiplier {multiplier:.2f}: Accuracy {caa_result.mcq_accuracy:.2f}, Average key prob {caa_result.average_key_prob:.2f}"
                )

    return caa_results


def run_load_extract_and_save(
    config: SteeringConfig,
):
    test_dataset = make_dataset(config.test_dataset_name, config.test_split_name)
    model_name = get_model_name(config.use_base_model, config.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)

    if len(config.layers) == 0:
        config.layers = list(range(model.config.num_hidden_layers))
    pipeline = Pipeline(model, tokenizer, formatter=LlamaChatFormatter())
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
    parser.add_argument("--datasets", type=str, default="")

    args = parser.parse_args()
    config = args.config

    if args.datasets:
        all_datasets = list_subset_of_datasets(args.datasets)
        for dataset_name in all_datasets:
            # Train and test on the same dataset
            print("Running on dataset: ", dataset_name)
            config.train_dataset_name = dataset_name
            config.test_dataset_name = dataset_name
            with EmptyTorchCUDACache():
                run_load_extract_and_save(config)
    else:
        with EmptyTorchCUDACache():
            run_load_extract_and_save(config)
