import os
import shutil
import subprocess

from repepo.data import get_dataset_dir

# Parameters
repo_url = "https://github.com/NoviScl/AmbigPrompt.git"
clone_dir = "/tmp/AmbigPrompt"
target_dir = get_dataset_dir()  # Replace with your local folder path
subdirectory = "testsets_ambiguous"


def clone_repo(repo_url, clone_dir):
    subprocess.run(["git", "clone", repo_url, clone_dir], check=True)


def copy_subdirectory(clone_dir, subdirectory, target_dir):
    src_path = os.path.join(clone_dir, subdirectory)
    dest_path = os.path.join(target_dir, subdirectory)
    shutil.copytree(src_path, dest_path)


def delete_repo(clone_dir):
    shutil.rmtree(clone_dir)


if __name__ == "__main__":
    # Execute the workflow
    try:
        clone_repo(repo_url, clone_dir)
        copy_subdirectory(clone_dir, subdirectory, target_dir)
        delete_repo(clone_dir)
        print("Workflow completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
