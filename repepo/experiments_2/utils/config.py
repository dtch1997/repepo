import os
from pathlib import Path

# Path to this experiment dir
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent.absolute()
# Dataset dir

WORK_DIR = Path(os.getenv("WORK_DIR", ".")).absolute()
DATASET_DIR = WORK_DIR / "datasets"

LOAD_IN_4BIT: int = int(os.getenv("LOAD_IN_4BIT", 0))
LOAD_IN_8BIT: int = int(os.getenv("LOAD_IN_8BIT", 0))