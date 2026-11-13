#!/bin/bash
#SBATCH -J ct_ssf_solar
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 -C V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t360
#SBATCH -qinferno
#SBATCH -o Report-ct_ssf_solar-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail
# Submit from the repository root, or set DSCP_RUNPATH explicitly.
RUNPATH="${DSCP_RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
module load anaconda3
conda activate "${CTSSF_ENV:-chronos}"

python -u -m sbatch.sbatch_run_ct_ssf.run_ct_ssf \
  --dataset solar --base-predictor chronos \
  --device cuda --seed "${CTSSF_SEED:-2026}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-4}" "$@"
