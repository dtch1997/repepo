import pandas as pd

from repepo.steering.utils.helpers import SteeringConfig
from repepo.core.evaluate import EvalResult
from dataclasses import asdict


def get_config_fields() -> list[str]:
    """Get the fields of the SteeringConfig dataclass"""
    return list(asdict(SteeringConfig()).keys())


def make_results_df(results: list[tuple[SteeringConfig, EvalResult]]):
    """Convert a list of results to a pandas DataFrame

    Each row in the dataframe corresponds to a single sample in the test set.
    """

    rows = []
    for config, result in results:
        row = {}
        row.update(asdict(config))
        row.update(result.metrics)
        # Sample-wise results
        for prediction in result.predictions:
            sample_row = row.copy()
            sample_row.update(
                {"test_positive_example": prediction.positive_output_prob.text}
            )
            sample_row.update(
                {"test_negative_example": prediction.negative_output_prob.text}
            )
            sample_row.update(prediction.metrics)
            rows.append(sample_row)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()
    return df
