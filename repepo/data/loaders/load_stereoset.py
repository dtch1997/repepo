from repepo.core import Dataset
from repepo.core.types import Example
from repepo.data import get_dataset_dir
from repepo.data.io import jload


FILE_NAME = "stereoset.json"


def load_stereoset() -> Dataset:
    """Load the pre-downloaded stereoset dataset from the data dir"""
    dataset_dir = get_dataset_dir()
    dataset_path = dataset_dir / FILE_NAME

    # TODO: should this auto-download if not found?
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"stereoset.json not found in {dataset_dir}. Please run `python -m repepo.data.generation.make_stereoset`"
        )

    with open(dataset_path, "r") as f:
        raw_dataset = jload(f)
    return [Example(**row) for row in raw_dataset]
