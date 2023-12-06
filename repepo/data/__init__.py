import functools
import pathlib
from typing import List

from repepo.variables import Environ
from repepo.core.types import Example

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


@functools.lru_cache(1)
def get_dataset(name: str):
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    example_dict_list = jload(_DATASETS[name])
    dataset = []
    for example_dict in example_dict_list:
        dataset.append(Example.from_dict(example_dict))
    return dataset
