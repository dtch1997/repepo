import torch
from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Protocol


def _mean_vector(diff_vecs: list[Tensor]) -> Tensor:
    """Compute the mean over the list of difference vectors."""
    return torch.mean(torch.cat(diff_vecs), dim=0)


def _norm_mean_vector(diff_vecs: list[Tensor]) -> float:
    """Compute the norm of the mean difference vector."""
    mean = _mean_vector(diff_vecs)
    return torch.norm(mean).item()


def _mean_norm(diff_vecs: list[Tensor]) -> float:
    """Compute the mean norm of the difference vectors."""
    return torch.mean(torch.stack([torch.norm(v) for v in diff_vecs])).item()


def _var_norm(diff_vecs: list[Tensor]) -> float:
    """Compute the variance of the norms of the difference vectors."""
    return torch.var(torch.stack([torch.norm(v) for v in diff_vecs])).item()


def _var_norm_scaled_by_mean_norm(diff_vecs: list[Tensor]) -> float:
    return _var_norm(diff_vecs) / _mean_norm(diff_vecs)


def _mean_euclidean_distance(diff_vecs: list[Tensor]) -> float:
    """Compute the average distance of the difference vectors to the mean vector."""
    mean_vector = _mean_vector(diff_vecs)
    distances = [diff_vec - mean_vector for diff_vec in diff_vecs]
    norm_vecs_to_mean = [torch.norm(vec) for vec in distances]
    return torch.mean(torch.stack(norm_vecs_to_mean)).item()


def _mean_euclidean_distance_scaled_by_norm_mean_vector(
    diff_vecs: list[Tensor],
) -> float:
    return _mean_euclidean_distance(diff_vecs) / _norm_mean_vector(diff_vecs)


def _mean_euclidean_distance_scaled_by_mean_norm(diff_vecs: list[Tensor]) -> float:
    return _mean_euclidean_distance(diff_vecs) / _mean_norm(diff_vecs)


def _mean_cosine_similarity(diff_vecs: list[Tensor]) -> float:
    """Compute the mean cosine similarity between each pair of difference vectors."""
    diff_vecs_tensor = torch.cat(diff_vecs)
    pairwise_cosine_sims = pairwise_cosine_similarity(
        diff_vecs_tensor, diff_vecs_tensor
    )
    return torch.mean(pairwise_cosine_sims).item()


def _mean_dot_product(diff_vecs: list[Tensor]) -> float:
    """Compute the mean dot product between each pair of difference vectors."""
    diff_vecs_tensor = torch.cat(diff_vecs)
    pairwise_dot_products = torch.einsum(
        "id,jd->ij", diff_vecs_tensor, diff_vecs_tensor
    )
    return torch.mean(pairwise_dot_products).item()


class Metric(Protocol):
    def __call__(self, diff_vecs: list[Tensor]) -> float:
        ...


def compute_difference_vectors(
    pos_acts: list[Tensor], neg_acts: list[Tensor]
) -> list[Tensor]:
    return [pos - neg for pos, neg in zip(pos_acts, neg_acts)]


class VarianceOfNormSimilarityMetric:
    name: str = "variance_of_norm_sim"

    def __call__(self, diff_vecs) -> float:
        return -1.0 * _var_norm_scaled_by_mean_norm(diff_vecs)


class EuclideanSimilarityMetric:
    name: str = "euclidean_sim"

    def __call__(self, diff_vecs: list[Tensor]) -> float:
        return -1.0 * _mean_euclidean_distance_scaled_by_mean_norm(diff_vecs)


class CosineSimilarityMetric:
    name: str = "cosine_sim"

    def __call__(self, diff_vecs: list[Tensor]) -> float:
        return _mean_cosine_similarity(diff_vecs)
