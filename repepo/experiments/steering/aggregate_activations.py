from repepo.experiments.steering.utils.helpers import (
    SteeringConfig,
    load_activations,
    save_concept_vectors,
)
from repepo.experiments.steering.utils.configs import list_configs

from steering_vectors.aggregators import (
    Aggregator,
    mean_aggregator,
    logistic_aggregator,
    pca_aggregator,
)
from steering_vectors.train_steering_vector import aggregate_activations

aggregators = {
    "mean": mean_aggregator,
    "logistic": logistic_aggregator,
    "pca": pca_aggregator,
}


def run_aggregate_activations(config: SteeringConfig):
    pos_acts = load_activations(config, "positive")
    neg_acts = load_activations(config, "negative")
    aggregator: Aggregator = aggregators[config.aggregator]()
    concept_vectors = aggregate_activations(pos_acts, neg_acts, aggregator)
    save_concept_vectors(config, concept_vectors)
    return concept_vectors


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(SteeringConfig, dest="config")
    parser.add_argument("--configs", type=str, default="")
    args = parser.parse_args()
    config = args.config

    if args.configs:
        configs = list_configs(
            args.configs, config.train_split_name, config.test_split_name
        )
        for config in configs:
            print(f"Running on dataset: {config.train_dataset_name}")
            run_aggregate_activations(config)

    else:
        run_aggregate_activations(config)
