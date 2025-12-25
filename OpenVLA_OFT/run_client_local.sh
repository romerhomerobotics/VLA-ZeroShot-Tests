#!/bin/bash
# Local launcher for the Isaac-side client (no Slurm needed).
# Run with: bash run_client_local.sh

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

exec ${PYTHON_BIN} client.py
