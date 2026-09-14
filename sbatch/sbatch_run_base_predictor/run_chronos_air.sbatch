#!/bin/bash
#SBATCH -J chronos_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 -C V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t60
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

RUNPATH=/storage/project/r-yxie77-0/jlee3541/projects/dscp
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${HF_HOME:-$RUNPATH/data/huggingface_cache}"
mkdir -p "$HF_HOME"

module load anaconda3
conda activate chronos

python -c 'import threadpoolctl; from threadpoolctl import ThreadpoolController; print("threadpoolctl:", threadpoolctl.__version__, threadpoolctl.__file__)'

python -u -m sbatch_run_base_predictor.run_chronos air-10 \
  --data-dir "$RUNPATH/data/bejing_air_quality" \
  --save-dir "$RUNPATH/data/air-10_prediction/chronos" \
  --window-length 100 \
  --prediction-length 10 \
  --device 0
