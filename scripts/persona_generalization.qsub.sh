#!/bin/bash
#$ -N persona_generalization             # Specify the job name
#$ -l h_vmem=64G           # Request 64GB of memory per job
#$ -l gpus=1               # Request 1 GPU
#$ -l gpu_type=L        # Specify the type of GPU

# Load the required modules
module load python3/3.11  # Load Python 3.11 module
module load cuda/11.2     # Load CUDA module, adjust version as necessary

# Navigate to the project directory
cd /home/ucabdc6/Scratch/repepo

# Create a Python virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -e . 

# Run the script
python repepo/experiments/persona_generalization.py

# Deactivate the virtual environment
deactivate
