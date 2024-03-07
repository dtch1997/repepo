from dataclasses import fields
from repepo.steering.run_experiment import run_experiment
from repepo.steering.utils.helpers import SteeringConfig, make_dataset


def get_sweep_variables(configs: list[SteeringConfig]) -> list[str]:
    """Find the fields which are swept over"""
    # For each field, construct a set of all values present in configs
    # Only fields which have more than 1 value are swept over
    sweep_variables = []
    for field in fields(SteeringConfig):
        values = set(getattr(config, field.name) for config in configs)
        if len(values) > 1:
            sweep_variables.append(field.name)
    return sweep_variables


def run_sweep(configs: list[SteeringConfig], sweep_name: str):
    for config in configs:
        run_experiment(config)


def test_dataset_exists(configs: list[SteeringConfig]):
    for config in configs:
        try:
            make_dataset(config.test_dataset)
        except FileNotFoundError:
            print(f"Dataset {config.test_dataset} not found")
            raise

        try:
            make_dataset(config.train_dataset)
        except FileNotFoundError:
            print(f"Dataset {config.train_dataset} not found")
            raise
