""" Environment variables """

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
    OutputDir = str(os.environ.get("OUTPUT_DIR", _get_project_dir() / "output"))
    DatasetDir = str(os.environ.get("DATASET_DIR", _get_project_dir() / "datasets"))
    LogDir = str(os.environ.get("LOG_DIR", _get_project_dir() / "logs"))

    # WandB
    WandbProject = os.environ.get("WANDB_PROJECT", "Rep-Eng")
    WandbEntity = os.environ.get("WANDB_ENTITY", "unknown")
    WandbGroup = os.environ.get("WANDB_GROUP", os.environ.get("USER", "default"))


# Model names
class Model:
    Pythia70m = "EleutherAI/pythia-70m"


if __name__ == "__main__":
    # Debugging
    print(_get_project_dir())
