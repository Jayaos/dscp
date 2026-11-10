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

Normal runs use the fixed `model.gamma` and `model.window_size` settings in
YAML. Each sequence has independent adaptive state during evaluation. Use the
[dedicated tuning jobs](#tuning-jobs) below to select those settings first.
Normal configs have no `tuning` section; older configs containing one are
rejected with guidance to the dedicated runner. Remove that section when
migrating an older fixed-run config. See the [ACI guide](../../baselines/aci/README.md)
for initialization and full/empty-set reporting details.

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

## Tuning jobs

These dedicated jobs use DSCP-QR's selection workflow: a shared config across
selected sequences and coverage levels, strict coverage-gap filtering, and
lowest mean Winkler among eligible candidates. ACI retains finite-interval
Winkler reporting and independent online states. No passing candidate means
no selected config. Tuning exports fixed configurations for the final-run jobs
above; the exported configurations contain no `tuning` section.

Submit from the repository root:

```bash
sbatch sbatch/sbatch_run_tuning/run_aci_air_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_aci_solar_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_aci_sapflux_tuning.sbatch
```

Each array runs LR, LSTM, and Chronos under the existing `dscp` environment,
with one CPU, 4 GB RAM, and four hours. It accepts the same environment/path
overrides as the fixed-default jobs. Extra arguments are forwarded to the
Python tuning CLI, for example `--sequence-key KEY` or `--top-k 5`.

Edit the dataset grid in `configs/aci_configs/aci_DATASET_tuning_config.yaml`
to change gamma candidates, window sizes, sequence count, or coverage threshold.
The supplied grids initialize from the first 50% of each selected sequence,
evaluate candidates over the following approximately 16%, and reserve the final
34% for testing. There are 16 candidates per predictor/dataset. The defaults
select 3 Air, 10 Solar, or 5 Sapflux sequences, as QR does.

Tuning outputs go to `results/tuning/aci_PREDICTOR_DATASET/`; rerunning the same
combination overwrites those trial files. Use a different `--save-dir` when
retaining multiple searches. Slurm logs use
`Report-aci-DATASET-tuning-JOBID_TASKID.out`. Tuning never runs the final test.

Inspect the ranking and use the winning fixed config for a separate final run:

```bash
python -m report.inspect_tuning_results results/tuning/aci_lr_air/tuning_results.pkl --top-k 3
# Replace 0007 with the winning trial index printed above.
python -m sbatch.sbatch_run_aci.run_aci results/tuning/aci_lr_air/trial_0007/resolved_config.yaml
```

To submit that selected config to Slurm, choose the corresponding predictor
array index (0=LR, 1=LSTM, 2=Chronos) and pass it to the existing ACI run job:

```bash
sbatch --array=0 sbatch/sbatch_run_aci/run_aci_air.sbatch \
  results/tuning/aci_lr_air/trial_0007/resolved_config.yaml
```

Final runs evaluate all sequences in the artifact, rebuilding their states
from the entire calibration prefix. See the [ACI tuning guide](../../baselines/aci/README.md#dscp-qr-style-hyperparameter-selection)
for the precise split, metric conventions, and local tuning command.
