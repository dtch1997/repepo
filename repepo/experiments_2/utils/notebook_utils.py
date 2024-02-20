from repepo.experiments_2.utils.helpers import (
    ConceptVectorsConfig,
    load_metrics,
    load_concept_vectors,
    list_subset_of_datasets,
    get_configs_for_datasets,
    LayerwiseMetricsDict,
    LayerwiseConceptVectorsDict
)

# TODO: Shouldn't import from experiments_2 script directly; 
# Need to move metrics into utils and import from there
from repepo.experiments_2.extract_concept_vectors import metrics 

import matplotlib.pyplot as plt

def list_all_metrics() -> list[str]:
    return list(metrics.keys())

def get_all_metrics(configs: list[ConceptVectorsConfig], metric_name: str) -> dict[str, LayerwiseMetricsDict]:
    all_metrics = {}
    for config in configs:
        name = config.train_dataset_name
        all_metrics[name] = load_metrics(config, metric_name)
    return all_metrics

def get_all_concept_vectors(configs: list[ConceptVectorsConfig]) -> dict[str, LayerwiseConceptVectorsDict]:
    all_concept_vectors = {}
    for config in configs:
        name = config.train_dataset_name
        all_concept_vectors[name] = load_concept_vectors(config)
    return all_concept_vectors

def plot_metric(ax: plt.Axes, all_metrics: dict[str, LayerwiseMetricsDict]) -> plt.Axes:
    for dataset_name, layerwise_metrics in all_metrics.items():
        ax.plot(
            layerwise_metrics.keys(), 
            layerwise_metrics.values(),
            label=dataset_name
        )
        ax.set_xlabel("Layer")
    ax.legend()
    return ax

def plot_all_metrics(configs: list[ConceptVectorsConfig], metric_name: str):
    all_metrics = get_all_metrics(configs, metric_name)
    fig, ax = plt.subplots()
    for i, (dataset_name, layerwise_metrics) in enumerate(all_metrics.items()):
        ax.plot(
            layerwise_metrics.keys(), 
            layerwise_metrics.values(),
            label=dataset_name
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Layer-wise {metric_name}")
    ax.legend()
    fig.tight_layout()