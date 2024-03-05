import pathlib
import random
from typing import List, TypeVar

from repepo.variables import Environ
from repepo.core.types import Example, Dataset, Completion
from repepo.translation.constants import LangOrStyleCode, LANG_OR_STYLE_MAPPING
from dataclasses import dataclass

from .io import jload


def get_raw_dataset_dir() -> pathlib.Path:
    return pathlib.Path(Environ.ProjectDir) / "raw_datasets"


def get_dataset_dir() -> pathlib.Path:
    return pathlib.Path(Environ.DatasetDir)


def get_all_json_filepaths(root_dir: pathlib.Path) -> List[pathlib.Path]:
    return list(root_dir.rglob("*.json"))


# Intentionally don't cache anything here, otherwise datasets don't be available after downloading
def _get_datasets(dataset_dir: pathlib.Path | None = None) -> dict[str, pathlib.Path]:
    datasets: dict[str, pathlib.Path] = {}
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    for path in get_all_json_filepaths(dataset_dir):
        datasets[path.stem] = path.absolute()
    return datasets


def list_datasets(dataset_dir: pathlib.Path | None = None) -> tuple[str, ...]:
    return tuple(_get_datasets(dataset_dir).keys())


@dataclass
class DatasetSplit:
    first_element: str
    second_element: str

    @staticmethod
    def from_str(split_str: str) -> "DatasetSplit":
        first, second = split_str.split(":")
        if first == "":
            first = "0"
        return DatasetSplit(first, second)

    def _parse_start_idx(self, length: int) -> int:
        element = self.first_element
        if element.endswith("%"):
            k = int(element[:-1]) * length // 100
        else:
            k = int(element)

        if k < 0 or k > length:
            raise ValueError(f"Invalid index: k={k} in {self}")
        return k

    def _parse_end_idx(self, length: int) -> int:
        element = self.second_element

        should_add = element.startswith("+")
        element = element.lstrip("+")

        if element.endswith("%"):
            k = int(element[:-1]) * length // 100
        else:
            k = int(element)

        if should_add:
            k = self._parse_start_idx(length) + k

        if k < 0 or k > length:
            raise ValueError(f"Invalid index: k={k} in {self}")

        return k

    def as_slice(self, length: int) -> slice:
        return slice(self._parse_start_idx(length), self._parse_end_idx(length))


@dataclass
class DatasetSpec:
    name: str
    split: str = ":100%"
    seed: int = 0

    def __repr__(self) -> str:
        return f"DatasetSpec(name={self.name},split={self.split},seed={self.seed})"


def _parse_split(split_string: str, length: int) -> slice:
    return DatasetSplit.from_str(split_string).as_slice(length)


def get_dataset(name: str, dataset_dir: pathlib.Path | None = None) -> Dataset:
    datasets = _get_datasets(dataset_dir)
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")

    example_dict_list = jload(datasets[name])
    dataset: Dataset = []
    for example_dict in example_dict_list:
        dataset.append(
            Example(
                positive=Completion(**example_dict["positive"]),
                negative=Completion(**example_dict["negative"]),
                meta=example_dict.get("meta", {}),
                steering_token_index=example_dict.get("steering_token_index", -1),
            )
        )
    return dataset


T = TypeVar("T")


def _shuffle_and_split(items: list[T], split_string: str, seed: int) -> list[T]:
    randgen = random.Random(seed)
    shuffled_items = items.copy()
    randgen.shuffle(shuffled_items)  # in-place shuffle
    split = _parse_split(split_string, len(shuffled_items))
    return shuffled_items[split]


def make_dataset(spec: DatasetSpec, dataset_dir: pathlib.Path | None = None):
    dataset = get_dataset(spec.name, dataset_dir)
    return _shuffle_and_split(dataset, spec.split, spec.seed)


@dataclass
class DatasetFilenameParams:
    base_name: str
    extension: str
    lang_or_style: LangOrStyleCode | None = None


def build_dataset_filename(
    base_name: str,
    extension: str = "json",
    lang_or_style: LangOrStyleCode | str | None = None,
) -> str:
    if lang_or_style is not None and lang_or_style not in LANG_OR_STYLE_MAPPING:
        raise ValueError(f"Unknown lang_or_style: {lang_or_style}")
    if extension.startswith("."):
        extension = extension[1:]
    if lang_or_style is None:
        return f"{base_name}.{extension}"
    return f"{base_name}--l-{lang_or_style}.{extension}"


def parse_dataset_filename(filename: str | pathlib.Path) -> DatasetFilenameParams:
    filepath = pathlib.Path(filename)
    if "--l-" in filepath.stem:
        base_name, lang_or_style = filepath.stem.split("--l-")
        if lang_or_style not in LANG_OR_STYLE_MAPPING:
            raise ValueError(f"Unknown lang_or_style: {lang_or_style}")
        return DatasetFilenameParams(
            base_name, extension=filepath.suffix, lang_or_style=lang_or_style
        )
    return DatasetFilenameParams(
        base_name=filepath.stem, extension=filepath.suffix, lang_or_style=None
    )
