import functools
import pathlib
import random
from typing import List, Any, NewType

from repepo.variables import Environ
from repepo.core.types import Example
from dataclasses import dataclass

from .io import jdump
from .io import jload


def get_dataset_dir() -> pathlib.Path:
    return pathlib.Path(Environ.DatasetDir)


def get_all_json_filepaths(root_dir: pathlib.Path) -> List[pathlib.Path]:
    return list(root_dir.rglob("*.json"))


_DATASETS = {}
for path in get_all_json_filepaths(get_dataset_dir()):
    _DATASETS[path.stem] = path.absolute()


@functools.lru_cache(1)
def list_datasets() -> tuple[str, ...]:
    return tuple(_DATASETS.keys())


@dataclass
class DatasetSpec:
    name: str
    split: str = ":100%"
    seed: int = 0  # unused atm


def parse_split(split_string, start_index, end_index) -> Any:
    # Check if the split string starts with a colon and ends with a percentage sign
    if split_string.startswith(":") and split_string.endswith("%"):
        try:
            # Remove the colon and percentage sign
            split_value = float(split_string[1:-1])
            # Calculate the start and end index of the split
            split_start = start_index
            split_end = start_index + int(
                (end_index - start_index + 1) * (split_value / 100)
            )
            return slice(split_start, split_end)

        except ValueError as e:
            # TODO: Write a nice error message
            raise ValueError(e)

    else:
        # Invalid format, return None
        raise NotImplementedError


@functools.lru_cache(1)
def get_dataset(name: str):
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    example_dict_list = jload(_DATASETS[name])
    dataset = []
    for example_dict in example_dict_list:
        dataset.append(Example(**example_dict))
    return dataset


def make_dataset(spec: DatasetSpec):
    dataset = get_dataset(spec.name)
    random.Random(spec.seed).shuffle(dataset)  # in-place shuffle
    split = parse_split(spec.split, 0, len(dataset))
    return dataset[split]
