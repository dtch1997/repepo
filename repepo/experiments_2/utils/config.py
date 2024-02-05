import os
from pathlib import Path

# Path to this experiment dir
EXPERIMENT_DIR = Path(__file__).resolve().parent.absolute()
# Dataset dir

WORK_DIR = Path(os.getenv("WORK_DIR", ".")).absolute()
DATASET_DIR = WORK_DIR / "datasets"
