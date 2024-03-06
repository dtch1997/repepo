""" Script to run a steering experiment. 

Example usage:
python repepo/experiments/run_experiment.py --config_path repepo/experiments/configs/sycophancy.yaml
"""

import matplotlib.pyplot as plt

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
)

from repepo.steering.extract_activations import extract_activations
from repepo.steering.aggregate_activations import aggregate_activations, get_aggregator
from repepo.steering.evaluate_steering_with_concept_vectors import (
    evaluate_steering_with_concept_vectors,
)
from repepo.steering.plot_results_by_layer import plot_results_by_layer


def run_plot_results(config):
    fig, ax = plt.subplots()
    results = load_results(config)
    plot_results_by_layer(ax, config, results)
    fig.tight_layout()

    save_path = f"results_{config.make_save_suffix()}.png"
    print("Saving results to: ", save_path)
    fig.savefig(save_path)


def run_experiment(config: SteeringConfig):
    # Set up pipeline
    model_name = get_model_name(config.use_base_model, config.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)
    formatter = get_formatter(config.formatter)
    pipeline = Pipeline(model, tokenizer, formatter=formatter)

    # Set up train dataset
    train_dataset = make_dataset(config.train_dataset_name, config.train_split_name)

    # Extract activations
    with EmptyTorchCUDACache():
        pos_acts, neg_acts = extract_activations(
            pipeline=pipeline,
            dataset=train_dataset,
            verbose=config.verbose,
        )

    # Aggregate activations
    aggregator = get_aggregator(config.aggregator)
    with EmptyTorchCUDACache():
        concept_vectors = aggregate_activations(
            pos_acts,
            neg_acts,
            aggregator,
            verbose=config.verbose,
        )

    # Evaluate steering with concept vectors
    test_dataset = make_dataset(config.test_dataset_name, config.test_split_name)
    with EmptyTorchCUDACache():
        results = evaluate_steering_with_concept_vectors(
            pipeline=pipeline,
            concept_vectors=concept_vectors,
            dataset=test_dataset,
            layers=config.layers,
            multipliers=config.multipliers,
            verbose=config.verbose,
        )

    # Save results
    save_results(config, results)


if __name__ == "__main__":
    import simple_parsing
    from pprint import pprint

    config = simple_parsing.parse(config_class=SteeringConfig, add_config_path_arg=True)
    pprint(config)
    run_experiment(config)
    run_plot_results(config)
