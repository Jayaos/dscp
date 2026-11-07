#!/bin/bash
#SBATCH --job-name=cop_table2_amzn
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --qos=inferno
#SBATCH --output=Report-cop-table2-amzn-%j.out
#SBATCH --error=Report-cop-table2-amzn-%j.out

set -euo pipefail

# Submit from the repository root, or set DSCP_RUNPATH explicitly.
DSCP_REPO_ROOT="${DSCP_RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$DSCP_REPO_ROOT/sbatch/sbatch_reproduce_cop_table2/run_amzn_prophet.py" ]]; then
  echo "Repository entrypoint missing under '$DSCP_REPO_ROOT'; set DSCP_RUNPATH to the repository root." >&2
  exit 2
fi
cd "$DSCP_REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if type module >/dev/null 2>&1; then
  module load "${COP_TABLE2_CONDA_MODULE:-anaconda3}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${COP_TABLE2_CONDA_ENV:-dscp}"

# All three methods run in one job by default. Optional CLI flags pass through.
exec python -B -u -m sbatch.sbatch_reproduce_cop_table2.run_amzn_prophet "$@"
