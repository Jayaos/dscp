#!/bin/bash
#SBATCH -J distmatch_solar
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0,1
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=192G
#SBATCH -t1440
#SBATCH --output=%x_%A_%a.out

set -euo pipefail

# Submit from the repository root, or set DSCP_RUNPATH explicitly.
RUNPATH="${DSCP_RUNPATH:-${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

if type module >/dev/null 2>&1; then
  module load "${DISTMATCH_CONDA_MODULE:-anaconda3}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DISTMATCH_ENV:-distmatch}"

# Array tasks 0/1/2 use LR/LSTM/Chronos forecasts, respectively.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '$TASK_ID'." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"

CONFIG_PATH="configs/distmatch_configs/distmatch_${BASE_PREDICTOR}_solar_config2.yaml"
# Every allocated node must see the repository, artifact, environment, and output directory.
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?Submit this script using sbatch.}}"
OUTPUT_ROOT="${DISTMATCH_OUTPUT_ROOT:-$RUNPATH/results/distmatch}"
# Resolve the trial datetime once so preparation, workers, and merge share it.
RUN_DATETIME="$(python -c 'from datetime import datetime; print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))')"
OUTPUT_DIR="$OUTPUT_ROOT/solar/$BASE_PREDICTOR/job_$JOB_ID/$RUN_DATETIME"
SHARD_COUNT="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-3}}"
if [[ ! "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Expected a positive Slurm node count; got '$SHARD_COUNT'." >&2
  exit 2
fi
WORKERS="${SLURM_CPUS_PER_TASK:-17}"
EXTRA_ARGS=(--num-cores "$WORKERS" --threads-per-worker 1 --output-dir "$OUTPUT_DIR")
if [[ -n "${DISTMATCH_SEED:-}" ]]; then
  EXTRA_ARGS+=(--seed "$DISTMATCH_SEED")
fi

# A shared manifest assigns different sequence keys to every allocated node.
python -u -m sbatch.sbatch_run_distmatch.run_distmatch \
  "$CONFIG_PATH" "${EXTRA_ARGS[@]}" "$@" --prepare-shards "$SHARD_COUNT"

# One coordinator per node starts up to WORKERS local sequence processes.
# SLURM_PROCID is expanded on each task's node and selects its shard index.
srun --nodes="$SHARD_COUNT" --ntasks="$SHARD_COUNT" --ntasks-per-node=1 --cpus-per-task="$WORKERS" \
  --kill-on-bad-exit=1 --export=ALL --label \
  bash -c 'exec python -u -m sbatch.sbatch_run_distmatch.run_distmatch "$@" --shard-index "$SLURM_PROCID"' \
  _ "$CONFIG_PATH" "${EXTRA_ARGS[@]}" "$@"

# set -e prevents publishing combined results if any node failed.
python -u -m sbatch.sbatch_run_distmatch.run_distmatch \
  "$CONFIG_PATH" "${EXTRA_ARGS[@]}" "$@" --merge-shards
