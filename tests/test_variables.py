import pathlib
from repepo.variables import Environ


def test_paths():
    project_dir = pathlib.Path(Environ.ProjectDir)
    assert project_dir.exists(), "Project directory does not exist"
    dataset_dir = pathlib.Path(Environ.DatasetDir)
    assert dataset_dir.exists(), "Dataset directory does not exist"
