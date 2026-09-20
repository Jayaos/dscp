#!/bin/bash
#SBATCH -J hopcpt_sapflux
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --gres=gpu:4 -C V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t720
#SBATCH -qinferno
#SBATCH -o Report-hopcpt-sapflux-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2: HopCPT on LR/LSTM/Chronos forecasts.
# GPU workers follow the Slurm allocation; change #SBATCH --gres or sbatch --gres.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

module load anaconda3
conda activate hopcpt

python -u -m sbatch_run_hopcpt.run_hopcpt_job sapflux \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${HOPCPT_OUTPUT_ROOT:-$RUNPATH/results/hopcpt}" \
  --seed "${HOPCPT_SEED:-2026}" \
  --num-gpus "${SLURM_GPUS_ON_NODE:?Slurm did not report the allocated GPU count}" \
  "$@"
