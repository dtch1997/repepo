""" Environment variables """

import os 
import pathlib

def _get_home_dir() -> pathlib.Path:
    return pathlib.Path.home().absolute()

def _get_project_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.absolute()

class Environ:
    HuggingfaceCacheDir = os.environ.get("HUGGINGFACE_CACHE_DIR", _get_home_dir() / '.huggingface')
    OutputDir = os.environ.get("OUTPUT_DIR", _get_project_dir() / 'output')
    DatasetDir = os.environ.get("DATASET_DIR", _get_project_dir() / 'datasets')
    LogDir = os.environ.get("LOG_DIR", _get_project_dir() / 'logs')

# Model names
class Model:
    Pythia70m = "EleutherAI/pythia-70m"

if __name__ == "__main__":
    # Debugging
    print(_get_project_dir())