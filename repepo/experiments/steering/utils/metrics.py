import torch
from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Callable

Metric = Callable[[list[Tensor]], float]


def _mean_agg(diff_vecs: list[Tensor]) -> Tensor:
    """Compute the mean over the list of difference vectors."""
    return torch.mean(torch.stack(diff_vecs), dim=0)


def _mean_norm(diff_vecs: list[Tensor]) -> float:
    """Compute the mean norm of the difference vectors."""
    return torch.mean(torch.stack([torch.norm(v) for v in diff_vecs])).item()


def _mean_var_scaled_by_norm_of_mean_of_diff_agg(diff_vecs: list[Tensor]) -> float:
    """Compute the mean variance of the difference vectors, scaled by the norm of the mean difference vector"""
    mean = _mean_agg(diff_vecs)
    vecs_to_mean = [diff_vec - mean for diff_vec in diff_vecs]
    mean_norm = torch.norm(mean)
    norm_vecs_to_mean = [torch.norm(vec) for vec in vecs_to_mean]
    return (torch.mean(torch.stack(norm_vecs_to_mean)) / mean_norm).item()


def _mean_var_scaled_by_mean_of_norm_of_diff_agg(diff_vecs: list[Tensor]) -> float:
    """Compute the mean variance of the difference vectors, scaled by the mean of the norms of the difference vectors."""
    mean = _mean_agg(diff_vecs)
    vecs_to_mean = [diff_vec - mean for diff_vec in diff_vecs]
    norm_vecs_to_mean = [torch.norm(vec) for vec in vecs_to_mean]
    return (torch.mean(torch.stack(norm_vecs_to_mean)) / _mean_norm(diff_vecs)).item()


def _mean_var_unscaled_agg(diff_vecs: list[Tensor]) -> float:
    """Compute the mean unscaled variance of the difference vectors."""
    mean = _mean_agg(diff_vecs)
    vecs_to_mean = [diff_vec - mean for diff_vec in diff_vecs]
    norm_vecs_to_mean = [torch.norm(vec) for vec in vecs_to_mean]
    return torch.mean(torch.stack(norm_vecs_to_mean)).item()


def _mean_pairwise_cosine_agg(diff_vecs: list[Tensor]) -> float:
    """Compute the mean cosine similarity between each pair of difference vectors."""
    diff_vecs_tensor = torch.stack(diff_vecs)
    pairwise_cosine_sims = pairwise_cosine_similarity(
        diff_vecs_tensor, diff_vecs_tensor
    )
    return torch.mean(pairwise_cosine_sims).item()


def _mean_pairwise_dot_product_agg(diff_vecs: list[Tensor]) -> float:
    """Compute the mean dot product between each pair of difference vectors."""
    diff_vecs_tensor = torch.stack(diff_vecs)
    pairwise_dot_products = torch.einsum(
        "id,jd->ij", diff_vecs_tensor, diff_vecs_tensor
    )
    return torch.mean(pairwise_dot_products).item()


metrics: dict[str, Metric] = {
    "mean_norm": _mean_norm,
    "var_mean_norm_diff": _mean_var_scaled_by_mean_of_norm_of_diff_agg,
    "var_norm_mean_diff": _mean_var_scaled_by_norm_of_mean_of_diff_agg,
    "unscaled_var": _mean_var_unscaled_agg,
    "pairwise_cosine": _mean_pairwise_cosine_agg,
    "pairwise_dot_product": _mean_pairwise_dot_product_agg,
}


def list_metrics():
    return list(metrics.keys())
