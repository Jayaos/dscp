# SplitCP experiment entry points

Run from the repository root using the existing `dscp` environment:

```bash
python -m sbatch.sbatch_run_split_cp.run_split_cp --dataset air --base-predictor lr --dry-run
python -m sbatch.sbatch_run_split_cp.run_split_cp --dataset air --base-predictor lr
python sbatch/sbatch_run_split_cp/run_split_cp.py configs/split_cp_configs/split_cp_lr_air_config.yaml
```

Presets cover `air`, `solar`, and `sapflux` with `lr`, `lstm`, and `chronos`
forecasts. Each preset uses the first 66% of each saved held-out sequence for
calibration and the remaining observations for testing. The split index is
`floor(T * calibration_ratio)`; compare exact test indices when matching another
baseline. There is no additional training/validation partition or random seed.

`--data-path`, `--output-dir`, and `--num-cores` (alias `--num_cores`) override
the YAML settings. Relative artifact and output paths resolve from the repository
root; a relative configuration filename resolves from the current directory.
Without a worker override, the YAML `num_cores` is retained. `--dry-run` validates
the configuration and reports artifact availability without loading data or
creating any output files. Default results go to
`results/split_cp_<predictor>_<dataset>/`.

Submit a dataset array job from the repository root:

```bash
sbatch sbatch/sbatch_run_split_cp/run_split_cp_air.sbatch
sbatch sbatch/sbatch_run_split_cp/run_split_cp_solar.sbatch
sbatch sbatch/sbatch_run_split_cp/run_split_cp_sapflux.sbatch
```

Array tasks 0, 1, and 2 select LR, LSTM, and Chronos. Set `DSCP_RUNPATH` to
override the repository location and `SPLIT_CP_ENV` to override the default
`dscp` conda environment. Additional arguments are forwarded to the Python CLI.
