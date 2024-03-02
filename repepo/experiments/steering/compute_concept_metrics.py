from repepo.experiments.steering.utils.helpers import (
    list_subset_of_datasets,
    get_configs_for_datasets,
    ConceptVectorsConfig,
    load_activations,
    compute_difference_vectors,
    save_metrics,
)
from repepo.experiments.steering.utils.metrics import list_metrics, get_metric


def run_load_extract_and_save(config: ConceptVectorsConfig, metric_names: list[str]):
    print("Running on dataset: ", config.train_dataset_name)
    positive_activations = load_activations(config, "positive")
    negative_activations = load_activations(config, "negative")
    difference_vectors = compute_difference_vectors(
        positive_activations, negative_activations
    )

    # Squeeze
    for layer_num, diff_vecs in difference_vectors.items():
        difference_vectors[layer_num] = [diff_vec.squeeze() for diff_vec in diff_vecs]

    for metric_name in metric_names:
        metric = get_metric(metric_name)
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

    metric_names = list_metrics()

    if args.datasets:
        all_datasets = list_subset_of_datasets(args.datasets)
        configs = get_configs_for_datasets(all_datasets, config.train_split_name)
        for config in configs:
            run_load_extract_and_save(config, metric_names=metric_names)
    else:
        run_load_extract_and_save(config, metric_names=metric_names)
