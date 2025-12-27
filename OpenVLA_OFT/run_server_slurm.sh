#!/bin/bash
#SBATCH --job-name=openvla-server
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --nodelist=ubem        
#SBATCH --gres=gpu:1  
#SBATCH --mail-user=aytac2003@gmail.com
#SBATCH --mail-type=ALL

set +u

# 2. Source Conda (Use the direct path, not .bashrc)
source ~/miniconda3/etc/profile.d/conda.sh

# 3. Activate the environment
conda activate openvla-oft

# 4. Re-enable strict variable checking
set -u

# ... (rest of the script)
echo "Environment activated. Python: $(which python)"
echo "N" | ${PYTHON_BIN:-python} server.py