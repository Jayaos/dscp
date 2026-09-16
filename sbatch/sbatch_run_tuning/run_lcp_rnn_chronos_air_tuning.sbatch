#!/bin/bash
#SBATCH -J rnn_lcp_air_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=gts-yxie77-paid
#SBATCH --gres=gpu:2 -C V100
#SBATCH --mem=8GB
#SBATCH -t480
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu


set -euo pipefail

# Optional: sbatch --gres=gpu:2 --cpus-per-task=4 --mem=16G <this-script> --num-gpus 2
# Without --num-gpus, the runner uses tuning.num_gpus from the tuning YAML.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
module load anaconda3
conda activate dscp


python -u sbatch/sbatch_run_tuning/run_local_cp_tuning.py \
  --base-config configs/lcp_configs/lcp_rnn_chronos_air_config.yaml \
  --grid-config configs/lcp_configs/lcp_rnn_air_tuning_config.yaml \
  --save-dir results/tuning/lcp_rnn_chronos_air \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026 \
  --num-gpus 2 \
  "$@"
