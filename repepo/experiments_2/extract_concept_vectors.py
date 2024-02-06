import torch
from torch import Tensor

from steering_vectors.aggregators import Aggregator

from repepo.experiments_2.utils.config import DATASET_DIR
from repepo.experiments_2.utils.helpers import (
    list_datasets,
    ConceptVectorsConfig,
    load_activation_differences,
    save_concept_vectors
)

def compute_concept_vectors(
    activations: dict[int, list[Tensor]],
    aggregator: Aggregator
) -> dict[int, Tensor]:
    concept_vectors: dict[int, Tensor] = {}
    for layer_num in activations.keys():
        diff_vecs = activations[layer_num]
        concept_vec = torch.mean(torch.stack(diff_vecs), dim=0)
        concept_vectors[layer_num] = concept_vec
    return concept_vectors

def run_load_extract_and_save(
    config: ConceptVectorsConfig,
    aggregator: Aggregator,
):
    print("Running on dataset: ", config.train_dataset_spec.name)
    difference_vectors = load_activation_differences(config)
    concept_vectors = compute_concept_vectors(
        difference_vectors,
        aggregator=aggregator
    )
    save_concept_vectors(config, concept_vectors)

if __name__ == "__main__":
    import simple_parsing
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ConceptVectorsConfig, dest="config")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--aggregator", type=str, default="mean")
    args = parser.parse_args()
    config = args.config

    if args.datasets:
        all_datasets = list_datasets(args.datasets)
        for dataset_name in all_datasets:
            config.train_dataset_spec.name = dataset_name
            run_load_extract_and_save(config, aggregator=args.aggregator)
    else:
        run_load_extract_and_save(config, aggregator=args.aggregator)