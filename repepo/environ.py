""" Environment variables """

import os 
import pathlib

def _get_home_dir() -> pathlib.Path:
    return pathlib.Path.home().absolute()

HUGGINGFACE_CACHE_DIR = os.environ.get("HUGGINGFACE_CACHE_DIR", _get_home_dir() / '.huggingface')