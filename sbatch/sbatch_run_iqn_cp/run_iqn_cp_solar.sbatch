#!/bin/bash
#SBATCH -J iqn_cp_solar
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0
#SBATCH --gres=gpu:1 -C V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t60
#SBATCH -qinferno
#SBATCH -o Report-iqn-cp-solar-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2: RNN on LR/LSTM/Chronos; 3/4/5: Transformer on LR/LSTM/Chronos.
# Select the legacy sampled head with --export=ALL,IQN_PREDICTION_HEAD=cosine_embedding.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

python -u -m sbatch_run_iqn_cp.run_iqn_cp_job solar \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --prediction-head "${IQN_PREDICTION_HEAD:-partially_monotonic}" \
  --output-root "${IQN_OUTPUT_ROOT:-$RUNPATH/results/iqn_cp}" \
  --seed "${IQN_SEED:-2026}"
