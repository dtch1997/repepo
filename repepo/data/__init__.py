import pathlib
import functools
from typing import Tuple
from .io import jload, jdump

from repepo.variables import Environ

def _get_current_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.absolute()

def get_dataset_dir() -> pathlib.Path:
    return Environ.DatasetDir

_DATASETS = {
    'stereoset': get_dataset_dir() / 'stereoset.json',
    'truthfulqa': get_dataset_dir() / 'truthfulqa.json'
}

def list_datasets() -> Tuple[str]:
    return list(_DATASETS.keys())

@functools.lru_cache(1)
def get_dataset(name: str):
    if name not in _DATASETS:
        raise ValueError(f'Unknown dataset: {name}')
    
    return jload(_DATASETS[name])