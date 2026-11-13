#!/bin/bash
#SBATCH -J lstm_air_save_cpu
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t120
#SBATCH -qinferno
#SBATCH -o Report-lstm-air-save-cpu-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
# All series train one global LSTM; parallelize its CPU tensor operations.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

python -u -m sbatch_run_base_predictor.run_lstm air-10 \
  --data-dir "$RUNPATH/data/bejing_air_quality" \
  --save-dir "$RUNPATH/data/air-10_prediction/lstm_model_saved" \
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
  --save-model \
  --device cpu
