from dataclasses import dataclass
import json
import pyrallis
from pyrallis import field
from repepo.algorithms.repe import RepeReadingControl
from repepo.data.make_dataset import DatasetSpec
from repepo.experiments.caa_repro.utils.helpers import (
    get_experiment_path,
    get_model_name,
    get_model_and_tokenizer,
    SteeringSettings,
)


@dataclass
class EvaluateCaaConfig:
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
    config: EvaluateCaaConfig,
) -> list[EvaluateCaaResult]:
    """Run in_distribution train and eval in CAA style in a single run"""
    model_name = get_model_name(
        config.settings.use_base_model, config.settings.model_size
    )
    model, tokenizer = get_model_and_tokenizer(model_name)

    repe_algo = RepeReadingControl(
        patch_generation_tokens_only=True,
        # CAA reads from position -2, since the last token is ")"
        read_token_index=-2,
        # CAA skips the first generation token, so doing the same here to match
        skip_first_n_generation_tokens=1,
        verbose=config.verbose,
    )


if __name__ == "__main__":
    config = pyrallis.parse(EvaluateCaaConfig)
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
