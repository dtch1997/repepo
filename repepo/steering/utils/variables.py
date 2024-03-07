import os
from pathlib import Path
from repepo.variables import Environ

WORK_DIR = Path(os.getenv("WORK_DIR", ".")).absolute()
DATASET_DIR = Path(Environ.DatasetDir)

LOAD_IN_4BIT: int = int(os.getenv("LOAD_IN_4BIT", 0))
LOAD_IN_8BIT: int = int(os.getenv("LOAD_IN_8BIT", 0))
