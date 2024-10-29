import pathlib
import torch
import pandas as pd

from typing import Literal
from steering_vectors import SteeringVector
from repepo.variables import Environ
from repepo.core.evaluate import EvalResult, EvalPrediction
from repepo.experiments.persona_generalization import (
    PersonaCrossSteeringExperimentResult,
)

EvalResultSweep = dict[float, EvalResult]  # A sweep over a multiplier
Model = Literal["llama7b", "qwen", "llama3_70b"]


def get_experiment_dir(model: Model) -> pathlib.Path:
    return (
        pathlib.Path(Environ.ProjectDir)
        / "experiments"
        / f"persona_generalization_{model}"
    )


def load_persona_cross_steering_experiment_result(
    dataset_name: str,
    experiment_dir: pathlib.Path = get_experiment_dir("qwen"),
) -> PersonaCrossSteeringExperimentResult:
    result_path = experiment_dir / f"{dataset_name}.pt"
    return torch.load(result_path)


def get_steering_vector(
    persona_cross_steering_experiment_result: PersonaCrossSteeringExperimentResult,
    steering_label: str = "baseline",
) -> SteeringVector:
    return persona_cross_steering_experiment_result.steering_vectors[steering_label]


def get_eval_result_sweep(
    persona_cross_steering_experiment_result: PersonaCrossSteeringExperimentResult,
    steering_label: str = "baseline",  # Label of the dataset used to train steering vector
    dataset_label: str = "baseline",  # Label of the dataset used to evaluate the steering vector
) -> EvalResultSweep:
    results = {}
    cross_steering_result = (
        persona_cross_steering_experiment_result.cross_steering_result
    )
    multipliers = list(cross_steering_result.steering.keys())

    dataset_idx = cross_steering_result.dataset_labels.index(dataset_label)
    steering_idx = cross_steering_result.steering_labels.index(steering_label)
    for multiplier in multipliers:
        results[multiplier] = cross_steering_result.steering[multiplier][dataset_idx][
            steering_idx
        ]
    # add the zero result
    results[0] = cross_steering_result.dataset_baselines[dataset_idx]
    return results


# Functions to make pandas dataframes
def eval_prediction_as_dict(
    prediction: EvalPrediction,
):
    dict = {}
    dict.update(prediction.metrics)

    if prediction.positive_output_prob is not None:
        dict["test_example.positive.text"] = prediction.positive_output_prob.text
    else:
        dict["test_example.positive.text"] = None
    if prediction.negative_output_prob is not None:
        dict["test_example.negative.text"] = prediction.negative_output_prob.text
    else:
        dict["test_example.negative.text"] = None
    return dict


def eval_result_as_df(
    eval_result: EvalResult,
) -> pd.DataFrame:
    # predictions
    rows = []
    for idx, pred in enumerate(eval_result.predictions):
        dict = eval_prediction_as_dict(pred)
        dict["test_example.idx"] = idx
        rows.append(dict)
    # TODO: metrics?
    return pd.DataFrame(rows)


def eval_result_sweep_as_df(
    eval_results: dict[float, EvalResult],
) -> pd.DataFrame:
    dfs = []
    for multiplier, result in eval_results.items():
        df = eval_result_as_df(result)
        df["multiplier"] = multiplier
        dfs.append(df)
    return pd.concat(dfs)
