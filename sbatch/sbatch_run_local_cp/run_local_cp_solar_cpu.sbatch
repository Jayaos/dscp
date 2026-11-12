#!/bin/bash
#SBATCH -J local_cp_solar_cpu
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH -t720
#SBATCH -qinferno
#SBATCH -o Report-local-cp-solar-cpu-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2: RNN on LR/LSTM/Chronos; use --array=0-5 to include Transformer.
# Local-CP model settings are inherited from configs/lcp_configs/ templates.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

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

output_args=()
if [[ -n "${LOCAL_CP_OUTPUT_ROOT:-}" ]]; then
  output_args=(--output-root "$LOCAL_CP_OUTPUT_ROOT")
fi

python -u -m sbatch_run_local_cp.run_local_cp_job solar \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --cpu-workers "${SLURM_CPUS_PER_TASK}" \
  "${output_args[@]}" \
  --seed "${LOCAL_CP_SEED:-2026}"
