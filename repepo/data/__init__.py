import functools
import pathlib
import random
import re
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
    # Define the regular expression pattern
    pattern = r"^(\d*):(\d*)%$"

    # Use regular expression to match and extract values
    match = re.match(pattern, split_string)

    # If there's a match, extract the start and end values
    if match:
        start_frac = int(match.group(1)) if match.group(1) else 0
        end_frac = int(match.group(2)) if match.group(2) else 100

        split_start = start_index + start_frac * (end_index - start_index) // 100
        split_end = start_index + end_frac * (end_index - start_index) // 100
        return slice(split_start, split_end)
    else:
        # Invalid format, return None
        raise ValueError(f"Parse string {split_string} not recognized")


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
