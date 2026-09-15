# DistMatch in DSCP

This baseline runs distribution matching on the signed residuals of an existing
point predictor: `residual = heldout_y - heldout_predictions`. It uses the
[DistMatch paper](https://openreview.net/pdf?id=SxBuTatzGe) and the official source
cloned in `dist_match_conformal` as its reference. Saved LR, LSTM, and Chronos
forecasts share the same input format; the baseline does not retrain them.

## Run

Create and activate the environment once:

```bash
conda env create -f envs/env-distmatch.yml
conda activate distmatch
```

From the repository root, validate the configuration and artifact path:

```bash
python -m sbatch.sbatch_run_distmatch.run_distmatch configs/distmatch_configs/distmatch_lr_air_config.yaml --dry-run
```

Run the experiment:

```bash
python -m sbatch.sbatch_run_distmatch.run_distmatch configs/distmatch_configs/distmatch_lr_air_config.yaml
```

`--dry-run` checks configuration and file availability without loading the
forecast artifact, importing numerical packages, creating output directories,
or fitting models. It returns an error for a missing forecast artifact. It does
not check the contents of a pickle; the experiment runner checks those when it
loads the artifact.

The presets cover LR, LSTM, and Chronos for air, solar, and Sapflow. The Sapflow
presets are templates: first generate the saved forecasts under
`data/sapflux-solo3-large/{lr,lstm,chronos}/` using the existing base-predictor
pipeline. `air-10` refers to the PM10 target; the air artifact can contain
multiple stations.

Relative artifact, output, and cache paths resolve from the repository root.
Use `--data-path`, `--output-dir`, or `--seed` for launch-specific overrides.
The launcher records the resolved settings in `launch_config.yaml` under the
output directory.

## CPU workers

The default presets set `num_cores: 4` and `threads_per_worker: 1`. Sequence
workers are separate processes. The CLI preserves the YAML worker count when
`--num-cores` is omitted; an explicit option overrides it:

```bash
python -m sbatch.sbatch_run_distmatch.run_distmatch configs/distmatch_configs/distmatch_lr_air_config.yaml --num-cores 2
```

Use `--threads-per-worker` to change each process's thread limit. Budget CPUs
for `num_cores * threads_per_worker` and memory for each simultaneous station.
The algorithm runs on CPU.

## Slurm jobs

Submit the dataset arrays from the repository root:

```bash
sbatch sbatch/sbatch_run_distmatch/run_distmatch_air.sbatch
sbatch sbatch/sbatch_run_distmatch/run_distmatch_solar.sbatch
sbatch sbatch/sbatch_run_distmatch/run_distmatch_sapflux.sbatch
```

Each array maps tasks `0`, `1`, and `2` to LR, LSTM, and Chronos. For example,
`sbatch --array=1 sbatch/sbatch_run_distmatch/run_distmatch_air.sbatch` runs only
air LSTM. Each task requests four CPUs, 16 GB of memory, and a 24-hour time limit.
Logs use `%x_%A_%a.out` (job name, array job ID, task ID).

For a single configuration, use the generic launcher:

```bash
sbatch sbatch/sbatch_run_distmatch/run_distmatch.sbatch configs/distmatch_configs/distmatch_lr_air_config.yaml
```

All launchers honor the YAML `num_cores` and `threads_per_worker`. Before loading
the model, the CLI checks that their product fits `SLURM_CPUS_PER_TASK`.
For example, eight workers with one thread each
require `sbatch --cpus-per-task=8 ...`; an insufficient allocation returns an
error without changing the configured worker count. The same check applies to
explicit CLI overrides inside a Slurm allocation. Supply cluster-specific
account/partition options to `sbatch` as needed.

The scripts load `anaconda3` when the cluster provides a `module` command, then
activate the `distmatch` environment. Create that environment first using the
YAML above. Override the module with `DISTMATCH_CONDA_MODULE`, the environment
with `DISTMATCH_ENV`, or the repository location with `DSCP_RUNPATH` (`RUNPATH`
is also accepted). `DISTMATCH_SEED` overrides the configured seed. Dataset
launchers forward additional arguments to the experiment CLI; the generic
launcher takes the configuration path first, followed by CLI arguments.

## Experiment protocol

Control the split and process count directly in the experiment configuration:

```yaml
num_cores: 4
threads_per_worker: 1
data:
  train_ratio: 0.50
  valid_ratio: 0.16
  test_ratio: 0.34
```

All ratios must be finite and sum to one. Training and test must be positive;
validation may be zero when using fixed settings without tuning. Boundaries
are `floor(T * train_ratio)` and that value plus `ceil(T * valid_ratio)`;
the remaining observations form test. No separate calibration split is used.
Use the same saved artifact and exact test target indices for comparisons
with other methods, whose default splits can differ.

The defaults partition each saved forecast suffix into 50% training, 16%
validation, and 34% test using DSCP's chronological boundaries. A history window
reduces the number of training examples while leaving the test boundary fixed.
The final run fits distribution trees on the training prefix, incorporates the
observed validation residuals, and then evaluates sequentially on test.

At each step, the interval uses previously observed residuals; the newly
observed residual is appended afterward. Default settings use 100-step residual
windows, a strict KS-statistic threshold of 0.1, ten distribution trees with
90% bootstrap sampling, and a fresh ten-tree, depth-two quantile forest for each
routed leaf. The beta search compares ten candidate endpoint pairs. The trees
keep their initial partitions and all observed leaf members.

The presets use residuals in their original units (`normalize: false`).
Independent random streams make repeated predictions reproducible and isolate
stations from worker scheduling. See the implementation's provenance notes for
adaptations to the original wrapper.

## Matching cache and memory

`matching.ks_block_size` controls the size of each exact KS computation block.
`matching.max_cache_memory_mb` controls when the boolean matching matrix spills
to a memory-mapped file. With `matching.cache_dir: null`, small matrices remain
in memory and larger matrices use temporary files. Set a persistent cache
directory to reuse compatible matching matrices across runs.

The matrix contains one entry per pair of training windows, so its storage and
initial matching work grow quadratically with the number of windows. This
setting is a matrix budget, not a limit on total worker memory. Leaf histories
and per-step quantile-forest fitting add memory and time. The implementation
keeps the full eligible training prefix; it does not silently cap calibration
history.

## Tune on validation

The supplied grid contains thresholds `[0.025, 0.05, 0.1]` and windows
`[25, 50, 100]`:

```bash
python -m sbatch.sbatch_run_tuning.run_distmatch_tuning --base-config configs/distmatch_configs/distmatch_lr_air_config.yaml --grid-config configs/distmatch_configs/distmatch_air_tuning_config.yaml --save-dir results/distmatch_tuning
```

Candidates use the same split boundaries and seed. Selection uses validation
coverage and Winkler score; final test evaluation is a separate command. If a
candidate passes the configured coverage condition, the tuner exports
`best_config.yaml`:

```bash
python -m sbatch.sbatch_run_distmatch.run_distmatch results/distmatch_tuning/best_config.yaml
```

To tune all three air predictors through Slurm:

```bash
sbatch sbatch/sbatch_run_tuning/run_distmatch_air_tuning.sbatch
```

This uses the same array mapping and CPU settings as the dataset launchers.
It checks configuration, artifact availability, and CPU allocation before
tuning. Results are separated by job and predictor under
`results/tuning/distmatch_air/job_<array_job_id>/<predictor>/`. Optional
environment variables are `DISTMATCH_GRID_CONFIG`,
`DISTMATCH_TUNING_OUTPUT_ROOT`, `DISTMATCH_TOP_K` (default `3`), and
`DISTMATCH_SEED`. The tuning job evaluates validation only; run the exported
`best_config.yaml` separately for final test results.

## Verify

```bash
python -m pytest tests/test_distmatch_model.py tests/test_distmatch_data_usage.py tests/test_distmatch_tuning.py tests/test_distmatch_cli.py
```

These checks cover KS/reference agreement, causality, train-only normalization,
validation isolation from test, serial/multicore equality, metrics and plotting,
and CLI/configuration behavior.

## Saved results

The runner writes `resolved_config.yaml`, `log.pkl`, `summary_results.pkl`, and
`run_metadata.yaml`. Logs contain original-scale interval endpoints and residual
offsets, exact target indices, coverage, width, and Winkler scores. Summaries
weight each series equally, including unequal-length series. Run metadata
records package versions, worker counts, and elapsed time; per-series metadata
records split boundaries, model/cache diagnostics, and training/replay/test
timings. Enabling plotting writes the standard interval PDFs under `plots/`.
