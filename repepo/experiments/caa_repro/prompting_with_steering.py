import torch
import json
import pyrallis

from pprint import pprint
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Iterable
from repepo.experiments.caa_repro.utils.helpers import (
    make_tensor_save_suffix,
    get_model_name,
    get_model_and_tokenizer,
    get_save_vectors_path,
    get_experiment_path,
    SteeringSettings,
)
from repepo.core.benchmark import EvalPrediction
from repepo.core.format import LlamaChatFormatter
from repepo.core.pipeline import Pipeline
from repepo.algorithms.repe import RepeReadingControl
from repepo.data.make_dataset import make_dataset
from repepo.core.types import Example
from repepo.core.evaluate import MultipleChoiceAccuracyEvaluator
from pyrallis import field
from steering_vectors.steering_vector import SteeringVector
from dataclasses import replace
from transformers.generation import GenerationConfig

save_vectors_path = get_save_vectors_path()
results_path = get_experiment_path() / "results"
analysis_path = get_experiment_path() / "analysis"


def evaluate_average_key_prob(predictions: List[EvalPrediction]) -> float:
    """Evaluates the average probability of the correct answer"""

    sum = 0
    count = 0
    for prediction in predictions:
        sum += prediction.get_normalized_correct_probs()
        count += 1
    return sum / count


def evaluate_pipeline(
    pipeline: Pipeline,
    example_iterator: Iterable[Example],
    generation_config: Optional[GenerationConfig] = None,
) -> List[EvalPrediction]:
    # evaluate
    predictions: list[EvalPrediction] = []
    requires_generation = False  # any([e.requires_generation for e in evaluators])
    requires_probs = True  # any([e.requires_probs for e in evaluators])

    for example in example_iterator:
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
    return predictions


@torch.no_grad()
def test_steering(
    layers: List[int], multipliers: List[float], settings: SteeringSettings
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    model_name = get_model_name(settings.use_base_model, settings.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)

    results = []

    # Only load the specified layers
    for layer_id in layers:
        layer_activation = torch.load(
            save_vectors_path
            / f"vec_layer_{make_tensor_save_suffix(layer_id, model_name)}.pt",
        ).to(model.device)

        steering_vector = SteeringVector(layer_activations={layer_id: layer_activation})
        dataset = make_dataset(settings.dataset_spec)

        for multiplier in multipliers:
            # Run steering with the specified layer and multiplier
            algorithm = RepeReadingControl(
                # NOTE: According to https://github.com/nrimsky/CAA/issues/2
                # the steering vector is currently applied at all token positions
                patch_generation_tokens_only=False,
                skip_reading=True,
                direction_multiplier=multiplier,
                override_vector=steering_vector,
            )
            pipeline = Pipeline(model, tokenizer, formatter=LlamaChatFormatter())
            # Run algorithm to create the hooks
            pipeline = algorithm.run(pipeline, dataset)

            predictions = evaluate_pipeline(
                pipeline,
                tqdm(dataset, desc=f"Testing layer {layer_id} multiplier {multiplier}"),
            )

            if settings.type == "in_distribution":
                evaluator = MultipleChoiceAccuracyEvaluator()
                mcq_accuracy = evaluator(predictions)["accuracy"]
                average_key_prob = evaluate_average_key_prob(predictions)
                results.append(
                    {
                        "layer_id": layer_id,
                        "multiplier": multiplier,
                        "mcq_accuracy": mcq_accuracy,
                        "average_key_prob": average_key_prob,
                        "type": settings.type,
                    }
                )

            elif settings.type == "out_of_distribution":
                raise NotImplementedError

            elif settings.type == "truthful_qa":
                # NOTE: for truthfulQA, we need to calculate MCQ accuracies by category
                raise NotImplementedError

    save_suffix = settings.make_result_save_suffix(None, None)
    with open(results_path / f"results_{save_suffix}.json", "w") as f:
        json.dump(results, f)


@dataclass
class PromptingWithSteeringConfig:
    """
    A single training sample for a steering vector.
    """

    layers: List[int] = field(default=[], is_mutable=True)
    multipliers: List[float] = field(default=[], is_mutable=True)
    settings: SteeringSettings = field(default=SteeringSettings(), is_mutable=True)


if __name__ == "__main__":
    results_path.mkdir(parents=True, exist_ok=True)
    config = pyrallis.parse(PromptingWithSteeringConfig)
    pprint(config)
    test_steering(config.layers, config.multipliers, config.settings)
