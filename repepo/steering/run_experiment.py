""" Defines a workflow to run a steering experiment. 

Example usage:
python repepo/steering/run_experiment.py --config_path repepo/experiments/configs/sycophancy.yaml
"""

import logging
import sys
import torch
import functools

from typing import cast
from pprint import pformat
from repepo.core.pipeline import Pipeline
from repepo.steering.utils.helpers import (
    SteeringConfig,
    EmptyTorchCUDACache,
    get_model_and_tokenizer,
    get_formatter,
    make_dataset,
    get_experiment_path,
    get_eval_result_path,
    save_eval_result,
    load_eval_result,
    get_activation_path,
    save_activation,
    load_activation,
    save_metric,
)

from repepo.steering.build_steering_training_data import (
    build_steering_vector_training_data,
)
from repepo.steering.concept_metrics import (
    VarianceOfNormSimilarityMetric,
    EuclideanSimilarityMetric,
    CosineSimilarityMetric,
    compute_difference_vectors,
)

from repepo.steering.utils.database import SteeringConfigDatabase

from steering_vectors.train_steering_vector import (
    extract_activations,
    SteeringVector,
    LayerType,
)

from repepo.steering.get_aggregator import get_aggregator
from repepo.steering.evaluate_steering_vector import (
    evaluate_steering_vector,
)


def get_logging_level_from_str(logging_level: str) -> int:
    if logging_level == "INFO":
        return logging.INFO
    elif logging_level == "DEBUG":
        return logging.DEBUG
    elif logging_level == "WARNING":
        return logging.WARNING
    elif logging_level == "ERROR":
        return logging.ERROR
    else:
        raise ValueError(f"Invalid logging level: {logging_level}")


# Cache to avoid adding multiple handlers to the logger
@functools.lru_cache(1)
def setup_logger(logging_level_str: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(__name__)
    logging_level = get_logging_level_from_str(logging_level_str)
    logger.setLevel(logging.DEBUG)

    # print to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def run_experiment(
    config: SteeringConfig,
    force_rerun: bool = False,
    logging_level: str = "INFO",
):
    # Set up logger
    logger = setup_logger(logging_level)
    logger.info(f"Running experiment with config: \n{pformat(config)}")

    # Set up database
    # NOTE: get_experiment_path().parent = '.../experiments'
    db_path = get_experiment_path().parent / "steering_config.sqlite"
    db = SteeringConfigDatabase(db_path=db_path)
    logger.info("Database contains {} entries".format(len(db)))

    # Insert config into database if it doesn't exist
    if db.contains_config(config):
        logger.info(f"Config {config} already exists in database.")
    else:
        db.insert_config(config)

    # Set up pipeline
    model, tokenizer = get_model_and_tokenizer(config.model_name)
    formatter = get_formatter(config.formatter)
    pipeline = Pipeline(model, tokenizer, formatter=formatter)

    # Set up train dataset
    train_dataset = make_dataset(config.train_dataset, config.train_split)
    steering_vector_training_data = build_steering_vector_training_data(
        pipeline, train_dataset, logger=logger
    )

    if get_activation_path(config.train_hash).exists() and not force_rerun:
        logger.info(f"Activations already exist for {config}. Skipping.")
        pos_acts, neg_acts = load_activation(config.train_hash)

    else:
        # Extract activations
        logging.info("Extracting activations.")
        with EmptyTorchCUDACache():
            pos_acts, neg_acts = extract_activations(
                pipeline.model,
                pipeline.tokenizer,
                steering_vector_training_data,
                layers=[config.layer],
                layer_type=cast(LayerType, config.layer_type),
                show_progress=True,
                move_to_cpu=True,
            )
            pos_acts = pos_acts[config.layer]
            neg_acts = neg_acts[config.layer]
            save_activation(config.train_hash, pos_acts, neg_acts)

        # Compute concept metrics
        concept_metrics = [
            VarianceOfNormSimilarityMetric(),
            EuclideanSimilarityMetric(),
            CosineSimilarityMetric(),
        ]
        diff_vecs = compute_difference_vectors(pos_acts, neg_acts)

        for metric in concept_metrics:
            metric_val = metric(diff_vecs)
            logger.debug(f"Metric {metric.name} | results: {metric_val}")
            save_metric(config.train_hash, metric.name, metric_val)

    # Aggregate activations
    aggregator = get_aggregator(config.aggregator)
    with EmptyTorchCUDACache():
        direction_vec = aggregator(torch.concat(pos_acts), torch.concat(neg_acts))
        steering_vector = SteeringVector(
            layer_activations={config.layer: direction_vec},
            layer_type=cast(LayerType, config.layer_type),
        )

    # Evaluate steering vector
    # We cache this part since it's the most time-consuming
    if get_eval_result_path(config.eval_hash).exists() and not force_rerun:
        logger.info(f"Results already exist for {config}. Skipping.")
        eval_result = load_eval_result(config.eval_hash)

    else:
        test_dataset = make_dataset(config.test_dataset, config.test_split)
        with EmptyTorchCUDACache():
            eval_results = evaluate_steering_vector(
                pipeline=pipeline,
                steering_vector=steering_vector,
                dataset=test_dataset,
                layers=[config.layer],
                multipliers=[config.multiplier],
                completion_template=config.test_completion_template,
                patch_generation_tokens_only=config.patch_generation_tokens_only,
                skip_first_n_generation_tokens=config.skip_first_n_generation_tokens,
                logger=logger,
            )
            assert len(eval_results) == 1, "Expected one result"
            eval_result = eval_results[0]
            save_eval_result(config.eval_hash, eval_result)


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser(add_config_path_arg=True)

    parser.add_arguments(SteeringConfig, dest="config")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--logging_level", type=str, default="INFO")
    args = parser.parse_args()
    config = args.config
    run_experiment(
        config, force_rerun=args.force_rerun, logging_level=args.logging_level
    )
