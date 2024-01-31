from dataclasses import dataclass
import json
from statistics import mean
import pyrallis
from pyrallis import field
from repepo.algorithms.repe import RepeReadingControl
from repepo.core.benchmark import Benchmark, evaluate_benchmark, train_benchmark
from repepo.core.evaluate import (
    MultipleChoiceAccuracyEvaluator,
    select_repe_layer_at_eval,
    set_repe_direction_multiplier_at_eval,
    update_completion_template_at_eval,
)
from repepo.core.format import LlamaChatFormatter
from repepo.data.make_dataset import make_dataset, DatasetSpec
from repepo.experiments.caa_repro.utils.helpers import (
    get_experiment_path,
    get_model_name,
    get_model_and_tokenizer,
    SteeringSettings,
)


@dataclass
class EvaluateTqaCaaConfig:
    experiment_name: str = field(default=None)
    layers: list[int] = field(default=[], is_mutable=True)
    multipliers: list[float] = field(default=[], is_mutable=True)
    settings: SteeringSettings = field(default=SteeringSettings(), is_mutable=True)
    train_dataset_spec: DatasetSpec = field(
        default=DatasetSpec(name="truthfulqa_caa", split=":80%"), is_mutable=True
    )
    test_dataset_spec: DatasetSpec = field(
        default=DatasetSpec(name="truthfulqa_caa", split="80:100%"), is_mutable=True
    )
    verbose: bool = True


@dataclass
class EvaluateCaaResult:
    layer_id: int
    multiplier: float
    mcq_accuracy: float
    average_key_prob: float
    type: str


def evaluate_tqa_caa(
    config: EvaluateTqaCaaConfig,
) -> list[EvaluateCaaResult]:
    """Run in_distribution train and eval in CAA style in a single run"""
    model_name = get_model_name(
        config.settings.use_base_model, config.settings.model_size
    )
    model, tokenizer = get_model_and_tokenizer(model_name)

    benchmark = Benchmark(
        name="CAA in-distribution",
        train_dataset=make_dataset(config.train_dataset_spec),
        test_dataset=make_dataset(config.test_dataset_spec),
        evaluators=[MultipleChoiceAccuracyEvaluator()],
    )

    repe_algo = RepeReadingControl(
        patch_generation_tokens_only=True,
        # CAA reads from position -2, since the last token is ")"
        read_token_index=-2,
        # CAA skips the first generation token, so doing the same here to match
        skip_first_n_generation_tokens=1,
        verbose=config.verbose,
    )

    trained_pipeline = train_benchmark(
        model,
        tokenizer,
        algorithms=[repe_algo],
        benchmark=benchmark,
        formatter=LlamaChatFormatter(),
    )

    results = []
    for layer_id in config.layers:
        for multiplier in config.multipliers:
            result = evaluate_benchmark(
                trained_pipeline,
                benchmark,
                eval_hooks=[
                    update_completion_template_at_eval(
                        "{prompt} My answer is {response}"
                    ),
                    set_repe_direction_multiplier_at_eval(multiplier),
                    select_repe_layer_at_eval(layer_id),
                ],
                verbose=config.verbose,
            )
            key_probs = [
                pred.get_normalized_correct_probs() for pred in result.predictions
            ]
            caa_result = EvaluateCaaResult(
                layer_id=layer_id,
                multiplier=multiplier,
                mcq_accuracy=result.metrics["accuracy"],
                average_key_prob=mean(key_probs),
                type="in_distribution",
            )
            if config.verbose:
                print(
                    f"Layer {layer_id}, multiplier {multiplier:.2f}: Accuracy {caa_result.mcq_accuracy:.2f}, Average key prob {caa_result.average_key_prob:.2f}"
                )
            results.append(caa_result)
    return results


if __name__ == "__main__":
    config = pyrallis.parse(EvaluateTqaCaaConfig)
    results = evaluate_tqa_caa(config)

    # save results to disk
    results_dicts = [res.__dict__ for res in results]
    save_suffix = config.settings.make_result_save_suffix(None, None)

    if config.experiment_name is None:
        config.experiment_name = (
            f"{config.train_dataset_spec.name}_{config.test_dataset_spec.name}"
        )

    results_path = get_experiment_path(config.experiment_name) / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"results_{save_suffix}.json", "w") as f:
        json.dump(results_dicts, f, ensure_ascii=False, indent=4)
