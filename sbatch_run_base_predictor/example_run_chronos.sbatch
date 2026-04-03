#!/bin/bash
#SBATCH --job-name=chronos-air10
#SBATCH --output=logs/chronos-air10-%j.out
#SBATCH --error=logs/chronos-air10-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /path/to/DSCP
mkdir -p logs

# Replace with your cluster environment setup.
# module load python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env

python -m sbatch_run_base_predictor.run_chronos air-10 \
  --window-length 100 \
  --prediction-length 10 \
  --device cuda \
  --save-dir data/air-10/chronos
