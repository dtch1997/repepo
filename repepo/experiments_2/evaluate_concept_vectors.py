import torch
import simple_parsing

from repepo.experiments_2.extract_concept_vectors import (
    ConceptVectorsConfig,
    get_experiment_path,
)


def load_concept_vectors_and_mean_relative_norms(
    config: ConceptVectorsConfig,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Load concept vectors and mean relative norms from disk"""
    experiment_path = get_experiment_path()
    result_save_suffix = config.make_result_save_suffix()
    vectors_save_dir = experiment_path / "vectors"
    concept_vectors = torch.load(
        vectors_save_dir / f"concept_vectors_{result_save_suffix}.pt"
    )
    mean_relative_norms = torch.load(
        vectors_save_dir / f"mean_relative_norms_{result_save_suffix}.pt"
    )
    return concept_vectors, mean_relative_norms


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ConceptVectorsConfig, dest="config")
    args = parser.parse_args()
    config = args.config

    concept_vectors, mean_relative_norms = load_concept_vectors_and_mean_relative_norms(
        config
    )
    print(concept_vectors)
