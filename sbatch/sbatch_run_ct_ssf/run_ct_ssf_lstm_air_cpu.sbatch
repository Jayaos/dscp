#!/bin/bash
#SBATCH -J ct_ssf_lstm_air_cpu
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t360
#SBATCH -qinferno
#SBATCH -o Report-ct_ssf_lstm_air_cpu-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail
# Submit from the repository root, or set DSCP_RUNPATH explicitly.
RUNPATH="${DSCP_RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
# Parallelize tensor operations; series are processed sequentially.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
module load anaconda3
conda activate "${CTSSF_ENV:-dscp}"

python -u -m sbatch.sbatch_run_ct_ssf.run_ct_ssf \
  --dataset air --base-predictor lstm \
  --device cpu --seed "${CTSSF_SEED:-2026}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-12}" "$@"
