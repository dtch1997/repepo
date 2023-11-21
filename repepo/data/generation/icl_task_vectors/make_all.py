import pathlib
from typing import Dict, List

from repepo.core.types import Example
from repepo.data import get_dataset_dir
from repepo.data import jdump
from repepo.data import jload
from repepo.data.generation.icl_task_vectors.make_knowledge_data import (
    prepare_knowledge_data,
)
from repepo.data.generation.icl_task_vectors.make_linguistic_data import (
    prepare_linguistic_data,
)
from repepo.data.generation.icl_task_vectors.make_translation_data import (
    prepare_translation_data,
)


def make_dataset_from_mapping(mapping: Dict[str, str]) -> List[Example]:
    examples = []
    for key, value in mapping.items():
        examples.append(Example(instruction="", input=key, output=value)._asdict())
    return examples


def get_all_json_filepaths(root_dir: pathlib.Path) -> List[pathlib.Path]:
    return list(root_dir.rglob("*.json"))


if __name__ in "__main__":
    prepare_knowledge_data()
    prepare_linguistic_data()
    prepare_translation_data()

    dataset_paths = get_all_json_filepaths(get_dataset_dir() / "icl_task_vectors")
    for path in dataset_paths:
        mapping = jload(path, "r")
        dataset: List[Example] = make_dataset_from_mapping(mapping)
        jdump(dataset, path, "w")
