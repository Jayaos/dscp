#!/bin/bash
#SBATCH -J chronos_sapflux_ctssf
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 -C V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t120
#SBATCH -qinferno
#SBATCH -o Report-chronos_sapflux_ctssf-%j.out
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

export HF_HOME="${HF_HOME:-$RUNPATH/data/huggingface_cache}"
mkdir -p "$HF_HOME"
REVISION_ARGS=()
if [[ -n "${CHRONOS_REVISION:-}" ]]; then
  REVISION_ARGS+=(--model-revision "$CHRONOS_REVISION")
fi
python -u -m sbatch.sbatch_run_base_predictor.run_chronos sapflux-solo3-large \
  --data-dir "$RUNPATH/data/sapflux/0.1.5/prepared" \
  --save-dir "$RUNPATH/data/sapflux-solo3-large/chronos_ctssf" \
  --window-length 100 --prediction-length 10 --device cuda \
  --save-ctssf "${REVISION_ARGS[@]}" "$@"
