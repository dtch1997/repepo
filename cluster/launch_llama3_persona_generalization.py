from git.repo import Repo
import shlex
from pathlib import Path
from repepo.experiments.get_datasets import get_all_prompts

with open(Path(__file__).parent / 'runner.yaml') as f:
    template = f.read()

repo = Repo(".")
commit_hash = str(repo.head.object.hexsha)
layer = 30 # found to be optimal in layer sweep

prompts = get_all_prompts()

for dataset_idx in range(len(get_all_prompts())):
    dataset_name = list(prompts.keys())[dataset_idx]
    command = [
        "python", 
        "repepo/experiments/persona_generalization.py",
        "--model_name=meta-llama/Meta-Llama-3.1-70B-Instruct",
        "--formatter_name qwen-chat-formatter",
        f"--layer={layer}",
        "--output_dir=/training/persona_generalization_llama3_70b", 
        # Hacky way to pass the dataset, for legacy reasons, sorry
        f"--sge_task_id={dataset_idx}"
    ]
    print(template.format(
        COMMAND=shlex.join(command),
        NAME=f"persona_generalization-{dataset_name}",
        IMAGE='ghcr.io/alignmentresearch/repepo:a26aee0-main',
        COMMIT_HASH=commit_hash,
        PRIORITY='normal-batch',
        CPU="4",
        MEMORY="200Gi",
        GPU="2",
        USER_ID=1001,
        GROUP_ID=1001,
        OMP_NUM_THREADS="\"4\"",
        TRAINING_MOUNT="/training"))
    print("---")