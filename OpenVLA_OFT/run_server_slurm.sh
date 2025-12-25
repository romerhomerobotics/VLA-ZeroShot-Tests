#!/bin/bash
# Simple Slurm launcher for the inference server.
# Submit with: sbatch run_server_slurm.sh

#SBATCH --job-name=openvla-server
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1                  # Run on 1 node
#SBATCH --ntasks=1                 # Run 1 task
#SBATCH --time=01:00:00            # Time limit (hrs:min:sec)

# -- Resource Allocation --
#SBATCH --partition=debug            # Submit to the 'debug' partition
#SBATCH --nodelist=romer3               # Specifically request the 'dl4' node
#SBATCH --gres=gpu:1  

# --- Email Notification Settings ---
#SBATCH --mail-user=aytac2003@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

# Ensure the logs directory exists to prevent sbatch error
mkdir -p logs

# Source bashrc to ensure conda is available (sometimes needed on non-interactive shells)
source ~/.bashrc
conda activate openvla-oft

# Optional: pin CUDA device (honors Slurm allocation)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Launch the FastAPI/uvicorn server
# Note: Ensure PYTHON_BIN is defined, otherwise replace with 'python'
exec ${PYTHON_BIN:-python} server.py