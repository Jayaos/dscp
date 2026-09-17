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

The normal templates set `data.calibration_ratio: 0.66`, reserving 66% of the
saved heldout forecasts for calibration. The remaining `1 - calibration_ratio`
(34%) is used for final testing; no separate test ratio is needed.
Hyperparameter validation is configured only in the tuning YAML through
`tuning.model_selection_valid_ratio`, which holds out a tail of the calibration
prefix. See the [baseline split documentation](../../baselines/kowcpi/README.md)
for the tuning protocol and migration from the old `train_ratio`/`valid_ratio` keys.

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

## Hyperparameter tuning arrays

Submit the dataset arrays from the repository root:

```bash
sbatch sbatch/sbatch_run_tuning/run_kowcpi_air_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_kowcpi_solar_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_kowcpi_sapflux_tuning.sbatch
```

Each array uses task IDs `0=lr`, `1=lstm`, `2=chronos`. Each task requests
12 CPUs, 48 GB memory, and 120 minutes in the `dscp` environment, matching the
existing KOWCPI tuning job. Independent sequences run in parallel within each
grid trial; the trials run sequentially. Job logs include both the array job
and task IDs.

The tuning dispatcher uses the same artifact paths and base templates as the
normal arrays, with `kowcpi_{dataset}_tuning_config.yaml` supplying the grid and
validation fraction. It writes a resolved base config and tuning artifacts to
`results/tuning/kowcpi/{dataset}/{predictor}/`. Final-test observations are
excluded from tuning, following the [baseline split protocol](../../baselines/kowcpi/README.md).

Preview any task locally without writing files or loading forecast artifacts:

```bash
python -m sbatch.sbatch_run_tuning.run_kowcpi_tuning_job solar --task-id 2 --dry-run
```

The preview needs only `omegaconf` beyond the Python standard library. To tune
only Chronos on Solar, for example:

```bash
sbatch --array=2 sbatch/sbatch_run_tuning/run_kowcpi_solar_tuning.sbatch
```

Set `KOWCPI_TUNING_OUTPUT_ROOT`, `KOWCPI_SEED`, or `KOWCPI_TOP_K` through
`sbatch --export=ALL,...` to override their defaults (`results/tuning/kowcpi`,
`2026`, and `3`). Set a distinct output root to retain repeated runs. Additional
dispatcher options may follow the script path, such as `--sequence-key`,
`--sequence-index`, `--grid-config`, or `--base-config`:

```bash
sbatch sbatch/sbatch_run_tuning/run_kowcpi_air_tuning.sbatch --sequence-index 3
```
