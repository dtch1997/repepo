import torch
from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Callable

from repepo.experiments.steering.utils.helpers import (
    list_subset_of_datasets,
    get_configs_for_datasets,
    ConceptVectorsConfig,
    load_activations,
    compute_difference_vectors,
    save_concept_vectors,
    save_metrics,
)

Aggregator = Callable[[list[Tensor]], Tensor]
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


def aggregate(
    activations: dict[int, list[Tensor]], aggregator: Aggregator
) -> dict[int, Tensor]:
    aggs: dict[int, Tensor] = {}
    for layer_num in activations.keys():
        diff_vecs = activations[layer_num]
        aggs[layer_num] = aggregator(diff_vecs)
    return aggs


def run_load_extract_and_save(config: ConceptVectorsConfig, metric_names: list[str]):
    print("Running on dataset: ", config.train_dataset_name)
    positive_activations = load_activations(config, "positive")
    negative_activations = load_activations(config, "negative")
    difference_vectors = compute_difference_vectors(
        positive_activations, negative_activations
    )

    # Squeeze
    for layer_num in difference_vectors.keys():
        difference_vectors[layer_num] = [
            diff_vec.squeeze(0) for diff_vec in difference_vectors[layer_num]
        ]

    concept_vectors = aggregate(difference_vectors, aggregator=_mean_agg)
    save_concept_vectors(config, concept_vectors)
    for metric_name in metric_names:
        metric = metrics[metric_name]
        metric_val = {
            layer_num: metric(diff_vecs)
            for layer_num, diff_vecs in difference_vectors.items()
        }
        save_metrics(config, metric_name, metric_val)  # type: ignore


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ConceptVectorsConfig, dest="config")
    parser.add_argument("--datasets", type=str, default="")
    # parser.add_argument("--aggregator", type=str, default="mean")
    args = parser.parse_args()
    config = args.config

    metric_names = list(metrics.keys())

    if args.datasets:
        all_datasets = list_subset_of_datasets(args.datasets)
        configs = get_configs_for_datasets(all_datasets, config.train_split_name)
        for config in configs:
            run_load_extract_and_save(config, metric_names=metric_names)
    else:
        run_load_extract_and_save(config, metric_names=metric_names)
