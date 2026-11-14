#!/bin/bash
#SBATCH -J qr_cp_lstm_saved_air_cpu
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t240
#SBATCH -qinferno
#SBATCH -o Report-qr-cp-lstm-saved-air-cpu-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Default task 1: RNN/LSTM. Use --array=4 for Transformer, or --array=1,4 for both.
# Reuse the matching configs/qr_cp_configs/ template, including its quantile head.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

case "${SLURM_ARRAY_TASK_ID}" in
  1|4) ;;
  *) echo "Saved-LSTM QR-CP jobs support only array tasks 1 and 4." >&2; exit 2 ;;
esac

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
# The existing QR-CP runner processes series sequentially; parallelize tensors.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

head_args=()
if [[ -n "${QR_HEAD_TYPE:-}" ]]; then
  head_args=(--head-type "$QR_HEAD_TYPE")
fi

python -u -m sbatch_run_qr_cp.run_qr_cp_job air \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --device cpu --num-threads "${SLURM_CPUS_PER_TASK}" \
  --data-path "$RUNPATH/data/air-10_prediction/lstm_model_saved/lstm_air-10_data.pkl" \
  "${head_args[@]}" \
  --output-root "${QR_OUTPUT_ROOT:-$RUNPATH/results/qr_cp_lstm_saved_cpu}" \
  --seed "${QR_SEED:-2026}"
