# KOWCPI Slurm jobs

Submit one array per dataset from the repository root on the configured cluster:

```bash
sbatch sbatch/sbatch_run_kowcpi/run_kowcpi_air.sbatch
sbatch sbatch/sbatch_run_kowcpi/run_kowcpi_solar.sbatch
sbatch sbatch/sbatch_run_kowcpi/run_kowcpi_sapflux.sbatch
```

The scripts follow [`../sbatch_run_qr_cp/`](../sbatch_run_qr_cp/): the same
account, Inferno QoS, four CPUs, 16 GB host memory, 10-minute time limit,
repository path, and notification address. They load `anaconda3` and activate
the `dscp` conda environment. KOWCPI runs on CPUs, so no GPU is requested.
Each task passes `SLURM_CPUS_PER_TASK` as `--num-cores` to process independent
sequences in parallel, with BLAS/OpenMP threads limited to one per worker.
Adjust resources with `sbatch --cpus-per-task=N --mem=SIZE --time=HH:MM:SS`
as needed for longer sequences.

## Array tasks and inputs

| Task ID | Base predictor |
| --- | --- |
| 0 | Linear regression (`lr`) |
| 1 | LSTM (`lstm`) |
| 2 | Chronos (`chronos`) |

For example, submit only the Chronos task with:

```bash
sbatch --array=2 sbatch/sbatch_run_kowcpi/run_kowcpi_air.sbatch
```

Generate the saved forecasts with the matching jobs under
[`../sbatch_run_base_predictor/`](../sbatch_run_base_predictor/) first.
Here, `P` is the selected predictor (`lr`, `lstm`, or `chronos`):

| Dataset | Required artifact, relative to the repository root |
| --- | --- |
| Air | `data/air-10_prediction/P/P_air-10_data.pkl` |
| Solar | `data/solar_prediction/P/P_nsdb-60m_data.pkl` |
| Sapflux | `data/sapflux-solo3-large/P/P_sapflux-solo3-large_data.pkl` |

## Configuration and outputs

Air and Solar use `configs/kowcpi_configs/kowcpi_chronos_air_config.yaml`.
Sapflux uses `configs/kowcpi_configs/kowcpi_lstm_sapflux_config.yaml`.
Each template supplies starting hyperparameters for all three predictors;
the dispatcher substitutes the selected artifact path and sets
`prediction_step: 1`. No additional hyperparameter tuning is performed.
The templates retain their normalization settings (off for Air/Solar, on
for Sapflux).

Each task writes to `results/kowcpi/{dataset}/{predictor}/`:

```text
resolved_config.yaml
log.pkl
summary_results.pkl
plots/
```

The default seed is `2026`. Set `KOWCPI_SEED` or `KOWCPI_OUTPUT_ROOT` when
submitting; `RUNPATH` overrides the repository path. Repeating a combination
reuses its output directory, so choose a separate output root to retain
another experiment:

```bash
sbatch --export=ALL,KOWCPI_SEED=17,KOWCPI_OUTPUT_ROOT=/path/to/results/kowcpi_seed17 \
  sbatch/sbatch_run_kowcpi/run_kowcpi_air.sbatch
```

## Preview a task

From the repository root, inspect the resolved configuration with:

```bash
python -m sbatch.sbatch_run_kowcpi.run_kowcpi_job air --task-id 2 --dry-run
```

The preview requires `omegaconf`, but does not write files or require Slurm
or prediction artifacts. The existing config-file launcher remains available:

```bash
python -m sbatch.sbatch_run_kowcpi.run_kowcpi \
  configs/kowcpi_configs/kowcpi_chronos_air_config.yaml --num-cores 4
```
