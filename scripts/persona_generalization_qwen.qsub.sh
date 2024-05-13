#!/bin/bash
#$ -N persona_generalization_qwen            # Specify the job name
#$ -l h_rt=72:00:00
#$ -l h_vmem=64G           # Request 64GB of memory per job
#$ -l gpu=1               # Request 1 GPU
#$ -ac allow=L        # Specify the type of GPU
#$ -o $HOME/logs
#$ -e $HOME/logs
#$ -t 1-40

nvidia-smi
umask 0077

set -e

# ensure we have the most up to date version of code
cd ${HOME}/repepo
git pull
git checkout fix_minor_experiment_script
git pull

# this only makes sense on David's cluster dir, sorry
source ${HOME}/Scratch/gpu-pyenv/bin/activate

# Run the script
PYTHON_PATH=. python3 -m repepo.experiments.persona_generalization \
    --layer 21 \
    --model_name Qwen/Qwen1.5-14B-Chat \
    --formatter_name qwen-chat-formatter \
    --output_dir ${HOME}/Scratch/persona_generalization_qwen \
    --sge_task_id $SGE_TASK_ID
