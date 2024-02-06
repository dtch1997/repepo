import torch
from torch import Tensor
from typing import Callable

Aggregator = Callable[[list[Tensor]], Tensor]
Metric = Callable[[list[Tensor]], float]

from repepo.experiments_2.utils.config import DATASET_DIR
from repepo.experiments_2.utils.helpers import (
    list_subset_of_datasets,
    ConceptVectorsConfig,
    load_activation_differences,
    save_concept_vectors,
    save_metrics
)

def _mean_agg(diff_vecs: list[Tensor]) -> Tensor:
    return torch.mean(torch.stack(diff_vecs), dim=0)

def _mean_norm(diff_vecs: list[Tensor]) -> float:
    return torch.mean(torch.stack([torch.norm(v) for v in diff_vecs])).item()

def _scaled_var_agg(diff_vecs: list[Tensor]) -> float:
    mean = _mean_agg(diff_vecs)
    vecs_to_mean = [diff_vec - mean for diff_vec in diff_vecs]
    # mean_norm = _mean_norm(diff_vecs)
    mean_norm = torch.norm(mean)
    norm_vecs_to_mean = [torch.norm(vec) for vec in vecs_to_mean]   
    return (torch.mean(torch.stack(norm_vecs_to_mean)) / mean_norm).item()

metrics: dict[str, Metric] = {
    "mean_norm": _mean_norm,
    "scaled_var": _scaled_var_agg
}

def aggregate(
    activations: dict[int, list[Tensor]],
    aggregator: Aggregator
) -> dict[int, Tensor]:
    aggs: dict[int, Tensor] = {}
    for layer_num in activations.keys():
        diff_vecs = activations[layer_num]
        aggs[layer_num] = aggregator(diff_vecs)
    return aggs

def run_load_extract_and_save(
    config: ConceptVectorsConfig,
    metric_names: list[str]
):
    print("Running on dataset: ", config.train_dataset_spec.name)
    difference_vectors = load_activation_differences(config)
    concept_vectors = aggregate(
        difference_vectors,
        aggregator=_mean_agg
    )
    save_concept_vectors(config, concept_vectors)
    for metric_name in metric_names:
        metric = metrics[metric_name]
        metric_val = {layer_num: metric(diff_vecs) for layer_num, diff_vecs in difference_vectors.items()}
        save_metrics(config, metric_name, metric_val) # type: ignore

if __name__ == "__main__":
    import simple_parsing
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ConceptVectorsConfig, dest="config")
    parser.add_argument("--datasets", type=str, default="")
    # parser.add_argument("--aggregator", type=str, default="mean")
    args = parser.parse_args()
    config = args.config

    metric_names = ["mean_norm", "scaled_var"]

    if args.datasets:
        all_datasets = list_subset_of_datasets(args.datasets)
        for dataset_name in all_datasets:
            config.train_dataset_spec.name = dataset_name
            run_load_extract_and_save(config, metric_names=metric_names)
    else:
        run_load_extract_and_save(config, metric_names=metric_names)