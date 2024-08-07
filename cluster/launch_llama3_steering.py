from git.repo import Repo
import shlex
from pathlib import Path

with open(Path(__file__).parent / 'runner.yaml') as f:
    template = f.read()


SWEEP_DATASETS = [
    "anti-immigration",
    "believes-abortion-should-be-illegal",
    "conscientiousness",
    "desire-for-acquiring-compute",
    "risk-seeking",
    "openness",
    "self-replication",
    "very-small-harm-justifies-very-large-benefit",
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
]

repo = Repo(".")
commit_hash = str(repo.head.object.hexsha)

for dataset in SWEEP_DATASETS:
    for layer in range(80):
        command = ["python", "repepo/experiments/sweep_llama3_70b_layers.py",
                   "--output_dir=/training/sweep2_llama3_70b/sweep_llama3_70b", f"--layer={layer}",
                   f"--datasets={dataset}"]
        print(template.format(
            COMMAND=shlex.join(command),
            NAME=f"{dataset[:10]}-{layer}",
            IMAGE='ghcr.io/alignmentresearch/repepo:a26aee0-main',
            COMMIT_HASH=commit_hash,
            PRIORITY='high-batch',
            CPU="4",
            MEMORY="200Gi",
            GPU="2",
            USER_ID=1001,
            GROUP_ID=1001,
            OMP_NUM_THREADS="\"4\"",
            TRAINING_MOUNT="/training"))
        print("---")
