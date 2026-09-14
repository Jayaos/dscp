#!/bin/bash
#SBATCH -J lstm_solar
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 -C V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t120
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

RUNPATH=/storage/project/r-yxie77-0/jlee3541/projects/dscp
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

module load anaconda3
conda activate dscp

python -u -m sbatch_run_base_predictor.run_lstm nsdb-60m \
  --data-dir "$RUNPATH/data/nsdb_2018-2020" \
  --save-dir "$RUNPATH/data/solar_prediction/lstm" \
  --predictor-train-ratio 0.33 \
  --window-length 100 \
  --embedding-dim 32 \
  --hidden-dim 32 \
  --num-layers 2 \
  --fit-train-ratio 0.9 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --max-epoch 50 \
  --early-stop 3 \
  --seed 2026 \
  --device cuda
