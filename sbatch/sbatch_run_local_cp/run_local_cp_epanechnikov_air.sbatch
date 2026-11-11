#!/bin/bash
#SBATCH -J local_cp_epanechnikov_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-5
#SBATCH --gres=gpu:1 -C V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t30
#SBATCH -qinferno
#SBATCH -o Report-local-cp-epanechnikov-air-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2: RNN on LR/LSTM/Chronos; 3/4/5: Transformer on LR/LSTM/Chronos.
# Model settings and bandwidth come from configs/lcp_epanechnikov_kernel_configs/.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

bandwidth_args=()
if [[ -n "${LOCAL_CP_BANDWIDTH:-}" ]]; then
  bandwidth_args=(--bandwidth "$LOCAL_CP_BANDWIDTH")
fi

output_args=()
if [[ -n "${LOCAL_CP_OUTPUT_ROOT:-}" ]]; then
  output_args=(--output-root "$LOCAL_CP_OUTPUT_ROOT")
fi

python -u -m sbatch_run_local_cp.run_local_cp_job air \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --kernel epanechnikov \
  "${bandwidth_args[@]}" \
  "${output_args[@]}" \
  --seed "${LOCAL_CP_SEED:-2026}"

