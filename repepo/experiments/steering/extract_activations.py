import torch
from repepo.core.types import Dataset
from repepo.core.pipeline import Pipeline
from repepo.experiments.steering.utils.helpers import (
    get_model_name,
    get_model_and_tokenizer,
    get_formatter,
    SteeringConfig,
    save_activations,
    make_dataset,
    EmptyTorchCUDACache,
)
from repepo.experiments.steering.utils.configs import list_configs
from steering_vectors.train_steering_vector import extract_activations


def _validate_train_dataset(dataset: Dataset):
    steering_token_index = dataset[0].steering_token_index
    for example in dataset:
        assert example.steering_token_index == steering_token_index


@torch.no_grad()
def run_extract_activations(
    config: SteeringConfig,
):
    model_name = get_model_name(config.use_base_model, config.model_size)
    model, tokenizer = get_model_and_tokenizer(model_name)
    formatter = get_formatter(config.formatter)
    pipeline = Pipeline(model, tokenizer, formatter=formatter)

    train_dataset = make_dataset(config.train_dataset_name, config.train_split_name)
    # Validate that all examples have the same steering token index
    _validate_train_dataset(train_dataset)
    read_token_index = train_dataset[0].steering_token_index

    repe_training_data = [
        (
            pipeline.build_full_prompt(example.positive),
            pipeline.build_full_prompt(example.negative),
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
        read_token_index=read_token_index,
        show_progress=True,
        move_to_cpu=True,
    )

    return pos_acts, neg_acts


def run_extract_and_save(config: SteeringConfig):
    with EmptyTorchCUDACache():
        pos, neg = run_extract_activations(config)
        save_activations(config, pos, "positive")
        save_activations(config, neg, "negative")


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(SteeringConfig, dest="config")
    parser.add_argument("--configs", type=str, default="")
    args = parser.parse_args()
    config = args.config

    if args.configs:
        configs = list_configs(
            args.configs, config.train_split_name, config.test_split_name
        )
        for config in configs:
            print(f"Running on dataset: {config.train_dataset_name}")
            run_extract_and_save(config)

    else:
        run_extract_and_save(config)
