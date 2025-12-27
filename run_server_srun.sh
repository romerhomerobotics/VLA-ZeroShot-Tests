#!/bin/bash
# Slurm batch launcher for the inference server. Submit with: sbatch run_server_srun.sh

#SBATCH --job-name=openvla-server
#SBATCH --partition=debug
#SBATCH --nodelist=romer3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=aytac2003@gmail.com
#SBATCH --mail-type=ALL


# Activate the OpenVLA-OFT environment (fixed path as requested)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate openvla-oft

exec python server.py
