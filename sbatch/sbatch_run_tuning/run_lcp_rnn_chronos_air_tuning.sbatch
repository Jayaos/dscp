#!/bin/bash
#SBATCH -J rnn_lcp_air_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --gres=gpu:1 -C V100
#SBATCH --mem=8GB
#SBATCH -t1440
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu


RUNPATH=/storage/project/r-yxie77-0/jlee3541/projects/dscp
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:$PYTHONPATH"
module load anaconda3
conda activate dscp


python sbatch/sbatch_run_tuning/run_local_cp_tuning.py \
  --base-config configs/lcp_configs/lcp_rnn_chronos_air_config.yaml \
  --grid-config configs/lcp_configs/lcp_rnn_air_tuning_config.yaml \
  --save-dir results/tuning/lcp_rnn_chronos_air \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026
