import torch
from torch import Tensor

from steering_vectors.aggregators import (
    Aggregator,
    mean_aggregator,
    logistic_aggregator,
    _uncentered_pca,
)


def pca_aggregator() -> Aggregator:
    """
    An aggregator that uses PCA to calculate a steering vector. This will always
    have norm of 1.
    """

    @torch.no_grad()
    def _pca_aggregator(pos_acts: Tensor, neg_acts: Tensor) -> Tensor:
        deltas = pos_acts - neg_acts
        # Note: Need to handle half-precision inputs
        deltas = deltas.to(torch.float32)
        neg_deltas = -1 * deltas
        vec = _uncentered_pca(torch.cat([deltas, neg_deltas]), k=1)[:, 0]
        # PCA might find the negative of the correct vector, so we need to check
        # that the vec aligns with most of the deltas, and flip it if not.
        sign = torch.sign(torch.mean(deltas @ vec))
        vec = vec.to(pos_acts.dtype)
        return sign * vec

    return _pca_aggregator


aggregators = {
    "mean": mean_aggregator,
    "logistic": logistic_aggregator,
    "pca": pca_aggregator,
}


def get_aggregator(name: str) -> Aggregator:
    """A wrapper around steering_vectors.aggregators.get_aggregator"""
    return aggregators[name]()
