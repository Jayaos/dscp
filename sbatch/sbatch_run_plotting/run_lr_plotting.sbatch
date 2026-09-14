#!/bin/bash
#SBATCH -J plot_lr
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t30
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

RUNPATH=/storage/project/r-yxie77-0/jlee3541/projects/dscp
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

python -u sbatch/sbatch_run_plotting/run_lr_plotting.py \
  --datasets air-10 nsdb-60m sapflux-solo3-large \
  --plot-len 1000 \
  --n-seqs 2
