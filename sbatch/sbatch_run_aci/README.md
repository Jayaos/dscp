# ACI runs

Run a preset from the repository root:

```bash
python -m sbatch.sbatch_run_aci.run_aci --dataset air --base-predictor lr --dry-run
python -m sbatch.sbatch_run_aci.run_aci --dataset air --base-predictor lr
python -m sbatch.sbatch_run_aci.run_aci configs/aci_configs/aci_lstm_solar_config.yaml
```

Both the module entrypoint and direct script execution work. Use `--data-path`,
`--output-dir`, and `--num-cores` to override those YAML settings. Relative
artifact/output paths resolve from the repository root. A dry run validates and
prints the resolved configuration without creating files or requiring the
forecast artifact to exist.

Nine presets cover Air, Solar, and Sapflux with LR, LSTM, and Chronos forecasts.
The calibration/test split is 66%/34%, with the same ratio-to-index rule as
NexCP. Presets target 90% coverage and use the original simple ACI update with
`gamma: 0.005`, a rolling `window_size: 1250`, and `initial_alpha: null`
(start at nominal miscoverage). Scores are absolute point-forecast residuals;
`quantile_method: linear` matches the author implementation's R quantiles.

The model settings are editable in YAML. Set `tuning.enabled: true` to select
`gamma` and `window_size` from their configured grids using only calibration
data. `tuning.validation_ratio` controls the calibration tail used to evaluate
candidates. Each sequence has independent adaptive state during evaluation.
See the [ACI guide](../../baselines/aci/README.md) for initialization, tuning,
and full/empty-set reporting details.

## Slurm

```bash
sbatch sbatch/sbatch_run_aci/run_aci_air.sbatch
sbatch sbatch/sbatch_run_aci/run_aci_solar.sbatch
sbatch sbatch/sbatch_run_aci/run_aci_sapflux.sbatch
```

Each job is an array with task 0 = LR, 1 = LSTM, and 2 = Chronos.
For one predictor, use e.g. `sbatch --array=1 ...`.
Defaults request one CPU, 4 GB RAM, and four hours. Override resources with
Slurm arguments as needed. Increase `num_cores` only when the job has a matching
CPU allocation; individual series still run chronologically.

The scripts retain the repository's cluster account and Inferno QoS, load
`anaconda3`, and activate `dscp`. Set `ACI_CONDA_MODULE` or `ACI_CONDA_ENV`
to use different installations. Submit from the repository root or set
`DSCP_RUNPATH`. The scripts do not configure email notifications.

Each run writes under `results/aci/DATASET/PREDICTOR/TIMESTAMP/`, as specified
by `saving_dir`. Slurm logs use `Report-aci-DATASET-JOBID_TASKID.out`.
