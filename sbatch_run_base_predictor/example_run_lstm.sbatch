#!/bin/bash
#SBATCH --job-name=lstm-air10
#SBATCH --output=logs/lstm-air10-%j.out
#SBATCH --error=logs/lstm-air10-%j.err
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

python -m sbatch_run_base_predictor.run_lstm air-10 \
  --predictor-train-ratio 0.33 \
  --window-length 100 \
  --embedding-dim 32 \
  --hidden-dim 32 \
  --num-layers 2 \
  --fit-train-ratio 0.9 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --max-epoch 10 \
  --early-stop 3 \
  --seed 2026 \
  --device cuda \
  --save-dir data/air-10/lstm
