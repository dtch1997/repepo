import pandas as pd
import numpy as np
from repepo.core.evaluate import EvalResult

from typing import NamedTuple
from dataclasses import asdict
from repepo.steering.plots.utils import (
    find_token_difference_position,
)

# A tuple of t


class SteeringVectorEvalTuple(NamedTuple):
    # A unique identifier for the steering vector.
    # E.g. the dataset it was trained on + other hparams.
    steering_vector_id: str
    eval_result: EvalResult


def make_results_df(results: list[EvalResult]):
    """Convert a list of results to a pandas DataFrame

    Each row in the dataframe corresponds to a single sample in the test set.
    """

    rows = []
    for result in results:
        row = {}
        row.update(
            {
                "steering_vector_id": result.steering_vector_id,
                "multiplier": result.multiplier,
            }
        )
        # Sample-wise results
        for prediction in result.predictions:
            sample_row = row.copy()
            # Add the text output
            sample_row.update(
                {
                    "test_positive_example.text": prediction.positive_output_prob.text,
                    "test_negative_example.text": prediction.negative_output_prob.text,
                }
            )
            # Add raw metrics
            sample_row.update(prediction.metrics)
            # Add other information about logits
            token_difference_position = find_token_difference_position(
                prediction.positive_output_prob, prediction.negative_output_prob
            )

            positive_token = prediction.positive_output_prob.token_probs[
                token_difference_position
            ]
            negative_token = prediction.negative_output_prob.token_probs[
                token_difference_position
            ]

            # Token info
            positive_token_as_dict = asdict(positive_token)
            negative_token_as_dict = asdict(negative_token)
            sample_row.update(
                {
                    f"test_positive_token.{k}": v
                    for k, v in positive_token_as_dict.items()
                }
            )
            sample_row.update(
                {
                    f"test_negative_token.{k}": v
                    for k, v in negative_token_as_dict.items()
                }
            )
            rows.append(sample_row)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()
    return df


def compute_steerability(
    eval_results: list[EvalResult],
    base_metric_name: str = "logit_diff",
):
    """Compute a steerability metric per example"""
    df = make_results_df(eval_results)
    assert (
        base_metric_name in df.columns
    ), f"Base metric {base_metric_name} not found in dataframe"

    # Group by examples
    fields_to_group_by = [
        "steering_vector_id",
        "test_positive_example.text",
    ]

    grouped = df.groupby(fields_to_group_by)

    def fit_linear_regression(df: pd.DataFrame):
        # Fit a linear regression of the base metric on the multiplier
        # Return the slope and error of the fit
        x = df["multiplier"].to_numpy()
        y = df[base_metric_name].to_numpy()
        (slope, intercept), residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
        # Return a dataframe with the slope and residuals
        return pd.DataFrame({"slope": [slope], "residual": [residuals.item()]})

    # Apply a linear-fit to each group using grouped.apply
    slopes = grouped.apply(fit_linear_regression, include_groups=False)
    df = df.merge(slopes, on=fields_to_group_by, how="left")
    return df
