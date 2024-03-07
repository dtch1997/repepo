from steering_vectors.aggregators import (
    Aggregator,
    mean_aggregator,
    logistic_aggregator,
    pca_aggregator,
)

aggregators = {
    "mean": mean_aggregator,
    "logistic": logistic_aggregator,
    "pca": pca_aggregator,
}


def get_aggregator(name: str) -> Aggregator:
    """A wrapper around steering_vectors.aggregators.get_aggregator"""
    return aggregators[name]()
