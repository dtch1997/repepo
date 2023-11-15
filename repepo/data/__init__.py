import pathlib
import functools
from typing import Tuple, List
from .io import jload, jdump

from repepo.variables import Environ

def get_dataset_dir() -> pathlib.Path:
    return pathlib.Path(Environ.DatasetDir)

def get_all_json_filepaths(root_dir: pathlib.Path) -> List[pathlib.Path]:
    return list(root_dir.rglob('*.json'))

_DATASETS = {}
for path in get_all_json_filepaths(get_dataset_dir()):
    _DATASETS[path.stem] = path.absolute()

@functools.lru_cache(1)
def list_datasets() -> Tuple[str]:
    return list(_DATASETS.keys())

@functools.lru_cache(1)
def get_dataset(name: str):
    if name not in _DATASETS:
        raise ValueError(f'Unknown dataset: {name}')
    
    return jload(_DATASETS[name])