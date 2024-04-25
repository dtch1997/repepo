#!/bin/bash
#$ -N persona_generalization             # Specify the job name
#$ -l h_rt=72:00:00
#$ -l h_vmem=64G           # Request 64GB of memory per job
#$ -l gpu=1               # Request 1 GPU
#$ -ac allow=L        # Specify the type of GPU

# Add locally installed executables to PATH
#  (e.g. pdm)
export PATH=$PATH:/home/$USER/.local/bin

# NOTE: Assume you have already cloned the repository and checked out correct branch
# Navigate to the project directory
cd /home/$USER/Scratch/repepo

# NOTE: Assume you have already installed pdm
pdm install

# Run the script
pdm run python repepo/experiments/persona_generalization.py --output_dir repepo/experiments/persona_generalization
