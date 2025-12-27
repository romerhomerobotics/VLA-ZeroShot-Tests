#!/bin/bash
# Local launcher for the Isaac-side client.
# Usage: bash run_client_local.sh


# Activate the OpenVLA-OFT environment (fixed path as requested)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate openvla-oft

exec python client.py
