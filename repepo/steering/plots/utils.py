import pandas as pd

from repepo.steering.utils.helpers import SteeringConfig
from repepo.core.evaluate import EvalResult, TextProbs
from dataclasses import asdict
import re


def is_linguistic_dataset(dataset_name: str) -> bool:
    return bool(re.match(r"^[DIE][0-9][0-9]", dataset_name))


def get_config_fields() -> list[str]:
    """Get the fields of the SteeringConfig dataclass"""
    return list(asdict(SteeringConfig()).keys())


def find_token_difference_position(text1: TextProbs, text2: TextProbs) -> int:
    """Find the first position where the two texts differ"""
    for i, (token1, token2) in enumerate(zip(text1.token_probs, text2.token_probs)):
        if token1.token_id != token2.token_id:
            return i
    return -1


def make_results_df(results: list[tuple[SteeringConfig, EvalResult]]):
    """Convert a list of results to a pandas DataFrame

    Each row in the dataframe corresponds to a single sample in the test set.
    """

    rows = []
    for config, result in results:
        row = {}
        row.update(asdict(config))
        row.update(
            {
                "train_hash": config.train_hash,
                "eval_hash": config.eval_hash,
            }
        )
        row.update(result.metrics)
        # Sample-wise results
        for prediction in result.predictions:
            assert prediction.positive_output_prob is not None
            assert prediction.negative_output_prob is not None
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
            # Ensure that the token difference position is valid
            # NOTE: doesn't work yet for linguistic datasets because prompt is empty...
            # Hardcode an exception for now
            if is_linguistic_dataset(config.train_dataset):
                token_difference_position = 0
            else:
                assert token_difference_position >= 0
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
