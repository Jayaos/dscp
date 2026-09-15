#!/bin/bash
#SBATCH --job-name=distmatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=Report-distmatch-%j.out

set -euo pipefail

# Submit from the repository root after activating the distmatch environment.
# Example: sbatch sbatch/sbatch_run_distmatch/run_distmatch.sbatch configs/distmatch_configs/distmatch_lr_air_config.yaml
RUNPATH="${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_PATH="${1:-configs/distmatch_configs/distmatch_lr_air_config.yaml}"
# Honor the YAML worker/thread counts. The CLI checks their product against
# SLURM_CPUS_PER_TASK before loading the numerical model or creating output.
python -u -m sbatch.sbatch_run_distmatch.run_distmatch "$CONFIG_PATH"
