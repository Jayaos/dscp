#!/bin/bash
#SBATCH -J hopcpt_solar
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --gres=gpu:2 -C V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t720
#SBATCH -qinferno
#SBATCH -o Report-hopcpt-solar-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2: HopCPT on LR/LSTM/Chronos forecasts.
# Two GPUs per task; override with --gres=gpu:1 <this-script> --num-gpus 1.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

module load anaconda3
conda activate hopcpt

python -u -m sbatch_run_hopcpt.run_hopcpt_job solar \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${HOPCPT_OUTPUT_ROOT:-$RUNPATH/results/hopcpt}" \
  --seed "${HOPCPT_SEED:-2026}" \
  --num-gpus 2 \
  "$@"
