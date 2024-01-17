import pathlib
import random
import re
from typing import List, TypeVar

from repepo.variables import Environ
from repepo.core.types import Example, Dataset
from dataclasses import dataclass

from .io import jload


def get_raw_dataset_dir() -> pathlib.Path:
    return pathlib.Path(Environ.ProjectDir) / "raw_datasets"


def get_dataset_dir() -> pathlib.Path:
    return pathlib.Path(Environ.DatasetDir)


def get_all_json_filepaths(root_dir: pathlib.Path) -> List[pathlib.Path]:
    return list(root_dir.rglob("*.json"))


# Intentionally don't cache anything here, otherwise datasets don't be available after downloading
def _get_datasets() -> dict[str, pathlib.Path]:
    datasets: dict[str, pathlib.Path] = {}
    for path in get_all_json_filepaths(get_dataset_dir()):
        datasets[path.stem] = path.absolute()
    return datasets


def list_datasets() -> tuple[str, ...]:
    return tuple(_get_datasets().keys())


@dataclass
class DatasetSpec:
    name: str
    split: str = ":100%"
    seed: int = 0


def _parse_split(split_string: str, length: int) -> slice:
    # Define the regular expression pattern
    pattern = r"^(\d*):(\d*)%$"

    # Use regular expression to match and extract values
    match = re.match(pattern, split_string)

    # If there's a match, extract the start and end values
    if match:
        start_frac = int(match.group(1)) if match.group(1) else 0
        end_frac = int(match.group(2)) if match.group(2) else 100

        split_start = start_frac * length // 100
        split_end = end_frac * length // 100
        return slice(split_start, split_end)
    else:
        # Invalid format, return None
        raise ValueError(f"Parse string {split_string} not recognized")


def get_dataset(name: str) -> Dataset:
    datasets = _get_datasets()
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")

    example_dict_list = jload(datasets[name])
    dataset: Dataset = []
    for example_dict in example_dict_list:
        dataset.append(Example(**example_dict))
    return dataset


T = TypeVar("T")


def _shuffle_and_split(items: list[T], split_string: str, seed: int) -> list[T]:
    randgen = random.Random(seed)
    shuffled_items = items.copy()
    randgen.shuffle(shuffled_items)  # in-place shuffle
    split = _parse_split(split_string, len(shuffled_items))
    return shuffled_items[split]


def make_dataset(spec: DatasetSpec):
    dataset = get_dataset(spec.name)
    return _shuffle_and_split(dataset, spec.split, spec.seed)
