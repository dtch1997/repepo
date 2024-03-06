from repepo.core.pipeline import Pipeline
from repepo.steering.utils.helpers import (
    SteeringConfig,
    EmptyTorchCUDACache,
    get_model_name,
    get_model_and_tokenizer,
    get_formatter,
    make_dataset,
    save_results,
)

from repepo.steering.extract_activations import extract_activations
from repepo.steering.aggregate_activations import aggregate_activations, get_aggregator
from repepo.steering.evaluate_steering_with_concept_vectors import (
    evaluate_steering_with_concept_vectors,
)

# Define configs to load

config = SteeringConfig(
    use_base_model=False,
    model_size="7b",
    train_dataset_name="sycophancy_train",
    train_split_name="train-dev",
    test_dataset_name="sycophancy_train",
    test_split_name="val-dev",
    formatter="llama-chat-formatter",
    aggregator="mean",
    layers=[0, 15, 31],
    multipliers=[-1, -0.5, 0, 0.5, 1],
    verbose=True,
)


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
    # import simple_parsing
    # parser = simple_parsing.ArgumentParser()
    # parser.add_arguments(SteeringConfig, dest="config")

    # config = parser.parse_args().config
    run_experiment(config)
