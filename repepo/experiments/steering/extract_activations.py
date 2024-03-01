import torch
from repepo.core.types import Dataset
from repepo.core.format import LlamaChatFormatter
from repepo.experiments.steering.utils.helpers import (
    get_model_name,
    get_model_and_tokenizer,
    ConceptVectorsConfig,
    save_activations,
    list_subset_of_datasets,
    get_configs_for_datasets,
    make_dataset,
    EmptyTorchCUDACache,
)

from steering_vectors.train_steering_vector import extract_activations


def _validate_train_dataset(dataset: Dataset):
    for example in dataset:
        assert example.steering_token_index == -2


@torch.no_grad()
def run_extract_activations(
    config: ConceptVectorsConfig,
):
    model_name = get_model_name(config.use_base_model, config.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)
    formatter = LlamaChatFormatter()

    train_dataset = make_dataset(config.train_dataset_name, config.train_split_name)
    _validate_train_dataset(train_dataset)

    repe_training_data = [
        (
            formatter.format_as_str(formatter.format_conversation(example.positive)),
            formatter.format_as_str(formatter.format_conversation(example.negative)),
        )
        for example in train_dataset
    ]

    _pos, _neg = repe_training_data[0]
    print(_pos)
    print(_neg)

    # Extract activations
    pos_acts, neg_acts = extract_activations(
        model,
        tokenizer,
        repe_training_data,
        read_token_index=-2,
        show_progress=True,
        move_to_cpu=True,
    )

    return pos_acts, neg_acts


def run_extract_and_save(config: ConceptVectorsConfig):
    with EmptyTorchCUDACache():
        pos, neg = run_extract_activations(config)
        save_activations(config, pos, "positive")
        save_activations(config, neg, "negative")


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ConceptVectorsConfig, dest="config")
    parser.add_argument("--datasets", type=str, default="")
    args = parser.parse_args()
    config = args.config

    if args.datasets:
        all_datasets = list_subset_of_datasets(args.datasets)
        configs = get_configs_for_datasets(all_datasets, config.train_split_name)
        for config in configs:
            print(f"Running on dataset: {config.train_dataset_name}")
            run_extract_and_save(config)

    else:
        run_extract_and_save(config)
