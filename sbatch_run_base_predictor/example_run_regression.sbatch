#!/bin/bash
#SBATCH --job-name=lr-air10
#SBATCH --output=logs/lr-air10-%j.out
#SBATCH --error=logs/lr-air10-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /path/to/DSCP
mkdir -p logs

# Replace with your cluster environment setup.
# module load python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env

python -m sbatch_run_base_predictor.run_regression air-10 \
  --train-ratio 0.33 \
  --past-window 100 \
  --prediction-step 1 \
  --save-dir data/air-10/lr
