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
    SteeringResult,
    EmptyTorchCUDACache,
    get_model_and_tokenizer,
    get_formatter,
    make_dataset,
    get_result_path,
    save_result,
    load_result,
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


# Cache to avoid adding multiple handlers to the logger
@functools.lru_cache(1)
def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # print to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def run_experiment(config: SteeringConfig):
    # Set up logger
    logger = setup_logger()
    logger.info(f"Running experiment with config: \n{pformat(config)}")

    # Set up database
    db = SteeringConfigDatabase()

    # Set up pipeline
    model, tokenizer = get_model_and_tokenizer(config.model_name)
    formatter = get_formatter(config.formatter)
    pipeline = Pipeline(model, tokenizer, formatter=formatter)

    # Set up train dataset
    train_dataset = make_dataset(config.train_dataset, config.train_split)
    steering_vector_training_data = build_steering_vector_training_data(
        pipeline, train_dataset, logger=logger
    )

    if get_activation_path(config.train_hash).exists():
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
    if get_result_path(config.eval_hash).exists():
        logger.info(f"Results already exist for {config}. Skipping.")
        result = load_result(config.eval_hash)

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
            result = SteeringResult(
                config_hash=config.eval_hash,
                mcq_acc=eval_result.metrics["mcq_acc"],
                logit_diff=eval_result.metrics["logit_diff"],
                pos_prob=eval_result.metrics["pos_prob"],
            )

        save_result(config.eval_hash, result)

    # Add config to database when all completed.
    db.insert_row(config)


if __name__ == "__main__":
    import simple_parsing

    config = simple_parsing.parse(config_class=SteeringConfig, add_config_path_arg=True)
    run_experiment(config)
