#!/bin/bash
#SBATCH -J local_cp_solar
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-5
#SBATCH --gres=gpu:1 -C V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t240
#SBATCH -qinferno
#SBATCH -o Report-local-cp-solar-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2: RNN on LR/LSTM/Chronos; 3/4/5: Transformer on LR/LSTM/Chronos.
# Local-CP model settings are inherited from configs/lcp_configs/ templates.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

python -u -m sbatch_run_local_cp.run_local_cp_job solar \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${LOCAL_CP_OUTPUT_ROOT:-$RUNPATH/results/local_cp}" \
  --seed "${LOCAL_CP_SEED:-2026}"
