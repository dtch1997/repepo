from statistics import mean
import torch
import json
import pyrallis

from pprint import pprint
from dataclasses import dataclass
from typing import List
from repepo.experiments.caa_repro.utils.helpers import (
    make_tensor_save_suffix,
    get_model_name,
    get_model_and_tokenizer,
    get_save_vectors_path,
    get_experiment_path,
    SteeringSettings,
)
from repepo.core.format import LlamaChatFormatter
from repepo.core.pipeline import Pipeline
from repepo.algorithms.repe import RepeReadingControl
from repepo.data.make_dataset import make_dataset
from repepo.core.evaluate import (
    MultipleChoiceAccuracyEvaluator,
    evaluate,
    set_repe_direction_multiplier_at_eval,
    update_completion_template_at_eval,
)
from pyrallis import field
from steering_vectors.steering_vector import SteeringVector

save_vectors_path = get_save_vectors_path()
results_path = get_experiment_path() / "results"
analysis_path = get_experiment_path() / "analysis"


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
        algorithm = RepeReadingControl(
            skip_reading=True,
            override_vector=steering_vector,
            patch_generation_tokens_only=True,
            # CAA skips the first generation token, so doing the same here to match
            skip_first_n_generation_tokens=1,
        )
        pipeline = Pipeline(model, tokenizer, formatter=LlamaChatFormatter())
        # Run algorithm to create the hooks
        pipeline = algorithm.run(pipeline, dataset)

        for multiplier in multipliers:
            # Run steering with the specified layer and multiplier

            result = evaluate(
                pipeline,
                dataset=dataset,
                tqdm_desc=f"Testing layer {layer_id} multiplier {multiplier}",
                evaluators=[MultipleChoiceAccuracyEvaluator()],
                eval_hooks=[
                    update_completion_template_at_eval(
                        "{prompt} My answer is {response}"
                    ),
                    set_repe_direction_multiplier_at_eval(multiplier),
                ],
            )

            if settings.type == "in_distribution":
                mcq_accuracy = result.metrics["accuracy"]
                key_probs = [
                    pred.get_normalized_correct_probs() for pred in result.predictions
                ]
                results.append(
                    {
                        "layer_id": layer_id,
                        "multiplier": multiplier,
                        "mcq_accuracy": mcq_accuracy,
                        "average_key_prob": mean(key_probs),
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
