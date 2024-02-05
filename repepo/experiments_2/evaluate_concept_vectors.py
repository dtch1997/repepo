import torch
import simple_parsing

from statistics import mean
from dataclasses import dataclass

from repepo.experiments_2.utils.helpers import (
    get_model_name,
    get_model_and_tokenizer,
)
from repepo.core.format import LlamaChatFormatter

from repepo.data.make_dataset import DatasetSpec, make_dataset
from repepo.core.types import Dataset
from repepo.core.pipeline import Pipeline
from repepo.core.evaluate import (
    MultipleChoiceAccuracyEvaluator,
    select_repe_layer_at_eval,
    set_repe_direction_multiplier_at_eval,
    update_completion_template_at_eval,
    evaluate
)
from repepo.algorithms.repe import SteeringHook
from steering_vectors import SteeringVector
from repepo.experiments_2.extract_concept_vectors import (
    ConceptVectorsConfig,
    get_experiment_path,
)

@dataclass
class EvaluateCaaResult:
    layer_id: int
    multiplier: float
    mcq_accuracy: float
    average_key_prob: float

def load_concept_vectors_and_mean_relative_norms(
    config: ConceptVectorsConfig,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Load concept vectors and mean relative norms from disk"""
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_result_save_suffix()
    vectors_save_dir = experiment_path / "vectors"
    concept_vectors = torch.load(
        vectors_save_dir / f"concept_vectors_{result_save_suffix}.pt"
    )
    mean_relative_norms = torch.load(
        vectors_save_dir / f"mean_relative_norms_{result_save_suffix}.pt"
    )
    return concept_vectors, mean_relative_norms

def evaluate_steering_with_concept_vectors(
    pipeline: Pipeline,
    concept_vectors: SteeringVector,
    dataset: Dataset,
    layers: list[int],
    multipliers: list[float],
    verbose: bool = False,
) -> list[EvaluateCaaResult]:
    caa_results = []

    # Create steering hook and add it to pipeline
    steering_hook = SteeringHook(
        steering_vector=concept_vectors,
        direction_multiplier=0,
        patch_generation_tokens_only=False,
        skip_first_n_generation_tokens=0,
        layer_config=None,
    )
    pipeline.hooks.append(steering_hook)

    for layer_id in config.layers:
        for multiplier in config.multipliers:
            pass
            # Run evaluate to get metrics
            result = evaluate(
                pipeline,
                dataset,
                eval_hooks = [
                    update_completion_template_at_eval(
                        "{prompt} My answer is {response}"
                    ),
                    set_repe_direction_multiplier_at_eval(multiplier),
                    select_repe_layer_at_eval(layer_id),
                ],
                evaluators = []
            )
            key_probs = [
                pred.get_normalized_correct_probs() for pred in result.predictions
            ]

            caa_result = EvaluateCaaResult(
                layer_id=layer_id,
                multiplier=multiplier,
                mcq_accuracy=result.metrics["accuracy"],
                average_key_prob=mean(key_probs),
            )
            caa_results.append(caa_result)
            if config.verbose:
                print(
                    f"Layer {layer_id}, multiplier {multiplier:.2f}: Accuracy {caa_result.mcq_accuracy:.2f}, Average key prob {caa_result.average_key_prob:.2f}"
                )

    # Remove steering hook
    pipeline.hooks[0].remove()
        
    return caa_results

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ConceptVectorsConfig, dest="config")
    parser.add_arguments(DatasetSpec, dest="test_dataset")
    parser.add_argument("layers", nargs="+", type=int)
    parser.add_argument("multipliers", nargs="+", type=float)

    args = parser.parse_args()
    config = args.config
    test_dataset = make_dataset(args.test_dataset) 

    model_name = get_model_name(config.use_base_model, config.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)
    pipeline = Pipeline(model, tokenizer, formatter=LlamaChatFormatter())

    concept_vectors, mean_relative_norms = load_concept_vectors_and_mean_relative_norms(
        config
    )
    pipeline = Pipeline(
        model=config.model,
        tokenizer=config.tokenizer,
        formatter=LlamaChatFormatter(),
    )

    results = evaluate_steering_with_concept_vectors(
        pipeline=pipeline,
        concept_vectors=concept_vectors,
        dataset=test_dataset,
        layers=args.layers,
        multipliers=args.multipliers,
        verbose=config.verbose,
    )
