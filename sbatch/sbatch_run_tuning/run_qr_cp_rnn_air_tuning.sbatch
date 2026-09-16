#!/bin/bash
#SBATCH -J rnn_qr_cp_air_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --gres=gpu:1 -C V100
#SBATCH --mem=8GB
#SBATCH -t720
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Choose base_predictor in the experiment YAML supplied via --base-config.

RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
module load anaconda3
conda activate dscp

python sbatch/sbatch_run_tuning/run_qr_cp_tuning.py \
  --base-config configs/qr_cp_configs/qr_rnn_chronos_air_config.yaml \
  --grid-config configs/qr_cp_configs/qr_rnn_air_tuning_config.yaml \
  --save-dir "results/tuning/qr_rnn_{base_predictor}_air" \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026
