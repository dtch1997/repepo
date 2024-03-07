""" Defines a workflow to run a steering experiment. 

Example usage:
python repepo/experiments/run_experiment.py --config_path repepo/experiments/configs/sycophancy.yaml
"""

import matplotlib.pyplot as plt
import logging
import sys

from pprint import pformat
from repepo.core.pipeline import Pipeline
from repepo.steering.utils.helpers import (
    SteeringConfig,
    EmptyTorchCUDACache,
    get_model_name,
    get_model_and_tokenizer,
    get_formatter,
    make_dataset,
    save_results,
    load_results,
    get_results_path,
)

from repepo.steering.build_steering_training_data import (
    build_steering_vector_training_data,
)
from steering_vectors.train_steering_vector import (
    extract_activations,
    aggregate_activations,
    SteeringVector,
)

from repepo.steering.get_aggregator import get_aggregator
from repepo.steering.evaluate_steering_vector import (
    evaluate_steering_vector,
)
from repepo.steering.plot_results_by_layer import plot_results_by_layer


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # print to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def run_plot_results(config):
    fig, ax = plt.subplots()
    results = load_results(config)
    plot_results_by_layer(ax, config, results)
    fig.tight_layout()

    save_path = f"results_{config.make_save_suffix()}.png"
    print("Saving results to: ", save_path)
    fig.savefig(save_path)


def run_experiment(config: SteeringConfig):
    logger = setup_logger()
    logger.info(f"Running experiment with config: \n{pformat(config)}")

    if get_results_path(config).exists():
        logger.info(f"Results already exist for {config}. Skipping.")
        return

    # Set up pipeline
    model_name = get_model_name(config.use_base_model, config.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)
    formatter = get_formatter(config.formatter)
    pipeline = Pipeline(model, tokenizer, formatter=formatter)

    # Set up train dataset
    train_dataset = make_dataset(config.train_dataset_name, config.train_split_name)
    steering_vector_training_data = build_steering_vector_training_data(
        pipeline, train_dataset, logger=logger
    )

    # Extract activations
    with EmptyTorchCUDACache():
        pos_acts, neg_acts = extract_activations(
            pipeline.model,
            pipeline.tokenizer,
            steering_vector_training_data,
            show_progress=True,
            move_to_cpu=True,
        )

    # TODO: compute intermediate metrics

    # Aggregate activations
    aggregator = get_aggregator(config.aggregator)
    with EmptyTorchCUDACache():
        agg_acts = aggregate_activations(
            pos_acts,
            neg_acts,
            aggregator,
        )
        steering_vector = SteeringVector(
            layer_activations=agg_acts,
            # TODO: make config option?
            layer_type="decoder_block",
        )

    # Evaluate steering vector
    test_dataset = make_dataset(config.test_dataset_name, config.test_split_name)
    with EmptyTorchCUDACache():
        results = evaluate_steering_vector(
            pipeline=pipeline,
            steering_vector=steering_vector,
            dataset=test_dataset,
            layers=config.layers,
            multipliers=config.multipliers,
            completion_template=config.test_completion_template,
            logger=logger,
        )

    # Save results
    save_results(config, results)


if __name__ == "__main__":
    import simple_parsing

    config = simple_parsing.parse(config_class=SteeringConfig, add_config_path_arg=True)
    run_experiment(config)
    run_plot_results(config)
