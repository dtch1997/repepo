from torch import Tensor
from steering_vectors.aggregators import (
    Aggregator,
    mean_aggregator,
    logistic_aggregator,
    pca_aggregator,
)
from steering_vectors.train_steering_vector import (
    aggregate_activations as _aggregate_activations,
)

aggregators = {
    "mean": mean_aggregator,
    "logistic": logistic_aggregator,
    "pca": pca_aggregator,
}


def get_aggregator(name: str) -> Aggregator:
    """A wrapper around steering_vectors.aggregators.get_aggregator"""
    return aggregators[name]()


def aggregate_activations(
    positive_activations: dict[int, list[Tensor]],
    negative_activations: dict[int, list[Tensor]],
    aggregator: Aggregator,
    verbose: bool = False,
):
    """A wrapper around steering_vectors.train_steering_vector.aggregate_activations"""
    return _aggregate_activations(
        positive_activations, negative_activations, aggregator
    )
