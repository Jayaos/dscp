#!/bin/bash
#SBATCH -J local_cp_lstm_saved_air_cpu
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t240
#SBATCH -qinferno
#SBATCH -o Report-local-cp-lstm-saved-air-cpu-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Task 1: RNN/LSTM; task 4: Transformer/LSTM, using CT-SSF's saved forecasts.
# Keep all Local-CP hyperparameters in the existing configs/lcp_configs/ templates.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

case "${SLURM_ARRAY_TASK_ID}" in
  1|4) ;;
  *) echo "Saved-LSTM Local-CP jobs support only array tasks 1 and 4." >&2; exit 2 ;;
esac

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
# Each worker processes one independent sequence with one CPU thread.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

python -u -m sbatch_run_local_cp.run_local_cp_job air \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --cpu-workers "${SLURM_CPUS_PER_TASK}" \
  --data-path "$RUNPATH/data/air-10_prediction/lstm_model_saved/lstm_air-10_data.pkl" \
  --output-root "${LOCAL_CP_OUTPUT_ROOT:-$RUNPATH/results/local_cp_lstm_saved_cpu}" \
  --seed "${LOCAL_CP_SEED:-2026}"
