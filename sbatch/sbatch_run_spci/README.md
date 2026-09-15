# SPCI Slurm jobs

Submit one array per dataset from the repository root on the configured cluster:

```bash
sbatch sbatch/sbatch_run_spci/run_spci_air.sbatch
sbatch sbatch/sbatch_run_spci/run_spci_solar.sbatch
sbatch sbatch/sbatch_run_spci/run_spci_sapflux.sbatch
```

The scripts follow [`../sbatch_run_hopcpt/`](../sbatch_run_hopcpt/): the same
account, Inferno QoS, four CPUs, 16 GB host memory, four-hour time limit,
repository path, and notification address. SPCI fits quantile random forests
on CPUs, so these jobs do not request a GPU. They activate the `spci` conda
environment defined in [`../../envs/env-spci.yml`](../../envs/env-spci.yml).
`LOKY_MAX_CPU_COUNT` caps the runner's `n_jobs=-1` at `SLURM_CPUS_PER_TASK`,
and BLAS/OpenMP threads are limited to one to avoid nested parallelism.
Adjust resources with `sbatch --cpus-per-task=N --mem=SIZE --time=HH:MM:SS`
as needed; runtime and memory depend on sequence count and length.

## Array tasks

| Task ID | Base predictor |
| --- | --- |
| 0 | Linear regression (`lr`) |
| 1 | LSTM (`lstm`) |
| 2 | Chronos (`chronos`) |

Each task calls the standard SPCI runner, fitting one forest per sequence.
To run only the LSTM-base combination:

```bash
sbatch --array=1 sbatch/sbatch_run_spci/run_spci_air.sbatch
```

## Required inputs

Generate the base-predictor artifacts with the matching jobs under
[`../sbatch_run_base_predictor/`](../sbatch_run_base_predictor/) first.
Here, `P` is the selected predictor (`lr`, `lstm`, or `chronos`):

| Dataset | Required artifact, relative to the repository root |
| --- | --- |
| Air | `data/air-10_prediction/P/P_air-10_data.pkl` |
| Solar | `data/solar_prediction/P/P_nsdb-60m_data.pkl` |
| Sapflux | `data/sapflux-solo3-large/P/P_sapflux-solo3-large_data.pkl` |

Air and Solar use the corresponding predictor's Air SPCI configuration
template. Sapflux uses `spci_lstm_sapflux_config.yaml` for all three
predictors, substituting the selected artifact path. These are starting
hyperparameters, with no additional tuning for Solar or Sapflux combinations.
The dispatcher sets `prediction_step: 1` to match the base-predictor jobs.

## Outputs and overrides

Each task writes to `results/spci/{dataset}/{predictor}/`:

```text
resolved_config.yaml
log.pkl
summary_results.pkl
plots/
```

The default seed is `2026`. The dispatcher seeds Python, NumPy, and PyTorch
before calling the SPCI runner. Set `SPCI_SEED` or `SPCI_OUTPUT_ROOT` when
submitting; set `RUNPATH` if the repository is elsewhere. Repeating a
combination reuses its output directory, so use a separate output root to
retain another experiment:

```bash
sbatch --export=ALL,SPCI_SEED=17,SPCI_OUTPUT_ROOT=/path/to/results/spci_seed17 \
  sbatch/sbatch_run_spci/run_spci_air.sbatch
```

## Preview a task

From the repository root, inspect the resolved configuration with:

```bash
python -m sbatch.sbatch_run_spci.run_spci_job air --task-id 1 --dry-run
```

The preview requires `omegaconf` and does not write files, train, or require
Slurm or prediction artifacts.
