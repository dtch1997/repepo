"""Environment variables"""

import os
import pathlib


def _get_home_dir() -> pathlib.Path:
    return pathlib.Path.home().absolute()


def _get_project_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.absolute()


class Environ:
    # Directories
    HuggingfaceCacheDir = str(
        os.environ.get(
            "HUGGINGFACE_HUB_CACHE", _get_home_dir() / ".cache/huggingface/hub"
        )
    )
    ProjectDir = str(_get_project_dir())
    TranslatedDatasetsDir = str(
        os.environ.get(
            "TRANSLATED_DATASETS_DIR", _get_project_dir() / "raw_datasets/translated"
        )
    )
    OutputDir = str(os.environ.get("OUTPUT_DIR", _get_project_dir() / "output"))
    DatasetDir = str(
        os.environ.get("DATASET_DIR", _get_project_dir() / "preprocessed_datasets")
    )
    LogDir = str(os.environ.get("LOG_DIR", _get_project_dir() / "logs"))

    # Google Cloud
    GcpProjectId = os.environ.get("GCP_PROJECT_ID", "repepo")


if __name__ == "__main__":
    # Debugging
    print(_get_project_dir())
