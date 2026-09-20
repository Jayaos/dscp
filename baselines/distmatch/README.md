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

## Progress and remaining time

Progress is enabled by default for both experiment runs and tuning. Each
sequence shows its name, stage, completed/total work, elapsed time, and ETA:

```text
DistMatch 'station_1' test:  40%|########            | 400/1000 [02:00 elapsed, ETA 03:00, 3.33step/s]
```

Stages are `matching`, `trees`, `replay` (when validation history is replayed),
and `test` or `validation`. The ETA refers to the **current stage**; during
prediction it estimates the remaining prediction time for that sequence.
It appears after work has completed and can change as the processing rate
changes. A cached matching matrix skips the matching stage.

Interactive terminals use a separate progress row per worker. Redirected
output, including SLURM logs, uses flushed lines roughly every 10 seconds of
completed work, plus stage starts and finishes. Set `show_progress: false`
at the top level of the YAML config to disable these progress displays.

## CPU workers

The presets configure worker counts with `num_cores` and use
`threads_per_worker: 1`. Sequence workers are separate processes.
The CLI preserves the YAML worker count when
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
air LSTM. The checked-in `#SBATCH` directives determine each launcher's active
array range, CPU, memory, and time requests. Logs use `%x_%A_%a.out` (job name,
array job ID, task ID).

For a single configuration, use the generic launcher:

```bash
sbatch sbatch/sbatch_run_distmatch/run_distmatch.sbatch configs/distmatch_configs/distmatch_lr_air_config.yaml
```

The generic, air, and Sapflow launchers honor the YAML `num_cores` and
`threads_per_worker`. The generic launcher's default 12-CPU request matches its
default LR Air preset (12 workers with one thread each). When selecting a preset
that requests more workers, override the Slurm CPU request or pass an explicit
worker override. Solar uses the allocated-node orchestration described below.
Before loading the model, the CLI checks that the worker/thread product fits
`SLURM_CPUS_PER_TASK` on each node.
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

### Solar across allocated nodes

The checked-in Solar launcher requests **three nodes, one task per node, and 17
CPUs per task**. Each task starts up to 17 independent sequence workers, with
one CPU thread per worker. A 50-sequence artifact is split into three disjoint
groups, allowing all 50 sequences to run concurrently when resources are
available:

```bash
# One three-node job for Solar LSTM.
sbatch --array=1 sbatch/sbatch_run_distmatch/run_distmatch_solar.sbatch

# LR only: --array=0. Chronos only: --array=2.
# The script's default array=0-1 submits both LR and LSTM, each on three nodes.
```

The script derives its shard count from `SLURM_JOB_NUM_NODES`, falling back to
`SLURM_NNODES` and then the checked-in three-node default. The same count is
passed to manifest preparation and to `srun` for both `--nodes` and `--ntasks`,
so every allocated node receives one shard. When overriding the node count,
also request the same task count. For example, two nodes with 25 workers each
can process a 50-sequence artifact concurrently:

```bash
sbatch --array=1 --nodes=2 --ntasks=2 --cpus-per-task=25 \
  sbatch/sbatch_run_distmatch/run_distmatch_solar.sbatch
```

Solar overrides YAML `num_cores` with `SLURM_CPUS_PER_TASK` and sets
`threads_per_worker=1`. The existing `--mem=192G` request applies **per node**;
adjust it for your partition and worker memory requirements. The repository,
forecast artifact, conda environment, and output directory must be accessible
from every allocated node through shared storage.

The batch task prepares one shard per allocated node, then `srun` starts shard
indices `0` through `N-1` on different nodes. Each sequence keeps its existing
seed, split, normalization, and evaluation procedure.
`srun --kill-on-bad-exit=1` stops the step if a node's task fails. The merge
runs only after every task succeeds and validates run identity, configuration,
artifact identity, and complete sequence membership. It writes one combined
log, summary, and set of plots in the usual format.

Outputs are isolated under
`results/distmatch_solar/job_<array_job_id>/<predictor>/`. Set
`DISTMATCH_OUTPUT_ROOT` to change the parent directory, or pass `--output-dir`
for a single predictor job. Preparation requires a fresh output directory to
avoid mixing results from different runs. Resubmitting normally uses a new job
ID; a requeued job with an existing output directory needs a fresh destination.
The manifest records ordered shard assignments, completion files live under
`shards/`, and merged metadata records total and per-shard worker counts.
Artifacts with other sequence counts are split evenly without dropping series.

The DistMatch CLI also exposes `--prepare-shards N`, `--shard-index I`, and
`--merge-shards` for manual orchestration. Use the same configuration and
overrides in all phases; these modes are mutually exclusive. An existing run's
saved `launch_config.yaml` can be used to retry an unfinished shard or merge
completed shards. Completed shards are not overwritten. If a task was killed
without cleanup, inspect its `.lock` file and confirm the process has stopped
before removing the lock and retrying that step. `--dry-run` validates arguments
and paths without preparing or executing shards.

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

Independent random streams make repeated predictions reproducible and isolate
stations from worker scheduling. See the implementation's provenance notes for
adaptations to the original wrapper.

### Normalization

The LR, LSTM, and Chronos presets enable target-based residual scaling:

```yaml
data:
  normalize_residual: true
```

`data.normalize_residual` defaults to `true` when omitted. Set it to `false`
to use raw residuals throughout matching and quantile-forest fitting. Raw-residual
runs ignore `train_y` and do not require normalization statistics.

When normalization is enabled, validation statistics use `heldout_y[:train_end]`; for final test,
they use `heldout_y[:validation_end]`, including observed validation history.
When an entry contains `train_y`, that history is prepended. Entries without
`train_y`, including existing Chronos artifacts, use only the observed held-out
prefix. This fallback does not use the entire held-out sequence. A provided
`train_y` must still be a valid, nonempty finite target sequence.

The mean and sample standard deviation (`ddof=1`) are frozen before the active
evaluation split; a constant target history uses scale one. At least two
historical targets are required. Validation tuning excludes validation targets
from its statistics, and final-test runs exclude test targets.

Both residual windows and quantile-forest targets are divided by this target
standard deviation. The target mean cancels in
`(y - mean) / std - (prediction - mean) / std`, so residuals are not centered
on the target mean. Predicted residual quantiles are multiplied by the standard
deviation before being added to the saved point forecasts. Interval endpoints,
widths, and Winkler scores remain in the original target units. Per-series
metadata records `normalize_residual` and `normalization.enabled`, with the
enabled normalization's source, target mean, standard deviation,
sample count, and held-out cutoff. Its `source` is
`train_y + heldout_y[:evaluation_start]` when training history is supplied, or
`heldout_y[:evaluation_start]` when the fallback is used.

The former `data.normalize` and `data.normalization_mode` keys are rejected
with a migration error. Replace `normalize: true` and
`normalization_mode: upstream_target` with `normalize_residual: true`, or
replace `normalize: false` with `normalize_residual: false`, removing both old
keys. The former `residual_inputs` mode has been removed; choose either target
standard-deviation scaling or raw residuals explicitly when migrating it.

Residual normalization follows the original target-scaling formula within the
saved-forecast workflow. Its historical population uses the available prefix;
the fallback omits any pre-forecast context that was not saved. It does not
retrain the base predictor on standardized features and targets, and therefore
does not reproduce the original forecasting pipeline. Artifact alignment and
split boundaries are unchanged.

The estimator core consumes the residual units supplied to `fit` and `observe`
without an additional normalization option. The data preparation and experiment
runner apply the configured scaling and restore predicted quantiles to original
units.

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

The air, solar, and sapflux tuning grids each search thresholds `[0.01, 0.1]`
with window length fixed at `100`, giving two combinations per base predictor.
The grids evaluate 3 air sequences,
10 solar sequences, and 5 sapflux sequences, respectively. For example,
run the air grid with the LR air base configuration:

```bash
python -m sbatch.sbatch_run_tuning.run_distmatch_tuning --base-config configs/distmatch_configs/distmatch_lr_air_config.yaml --grid-config configs/distmatch_configs/distmatch_air_tuning_config.yaml --save-dir results/distmatch_tuning
```

Use the matching dataset's base configuration and tuning grid for solar or
sapflux. Candidates use the same split boundaries and seed. All three grids
set `delta_threshold: -0.01`: coverage must exceed nominal coverage minus 0.01
for every selected sequence and confidence level. The eligible candidate with
the lowest mean validation Winkler score is exported as `best_config.yaml`.
If no candidate qualifies, no best configuration is exported. Final test
evaluation is a separate command:

```bash
python -m sbatch.sbatch_run_distmatch.run_distmatch results/distmatch_tuning/best_config.yaml
```

To tune LR, LSTM, and Chronos through Slurm for each dataset:

```bash
sbatch sbatch/sbatch_run_tuning/run_distmatch_air_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_distmatch_solar_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_distmatch_sapflux_tuning.sbatch
```

Each launcher uses its dataset's base configurations and tuning grid. Array
tasks `0`, `1`, and `2` select LR, LSTM, and Chronos, respectively, with four
CPUs per task. The launcher checks configuration, artifact availability, and
CPU allocation before tuning. Results are separated by dataset, job, and
predictor under
`results/tuning/distmatch_<dataset>/job_<array_job_id>/<predictor>/`. Optional
environment variables are `DISTMATCH_GRID_CONFIG`,
`DISTMATCH_TUNING_OUTPUT_ROOT`, `DISTMATCH_TOP_K` (default `3`), and
`DISTMATCH_SEED`. The tuning job evaluates validation only; run the exported
`best_config.yaml` separately for final test results.

## Verify

```bash
python -m pytest tests/test_distmatch_model.py tests/test_distmatch_data_usage.py tests/test_distmatch_normalization.py tests/test_distmatch_tuning.py tests/test_distmatch_cli.py tests/test_distmatch_progress.py tests/test_distmatch_distributed.py tests/test_distmatch_solar_launch.py
```

These checks cover KS/reference agreement, causality, normalization boundaries,
validation isolation from test, serial/multicore equality, metrics and plotting,
CLI/configuration behavior, merged shard equivalence, incomplete-run rejection,
and the Solar launch sequence with mocked Slurm commands. A real multi-node
Slurm submission must be verified on the target cluster.

## Saved results

The runner writes `resolved_config.yaml`, `log.pkl`, `summary_results.pkl`, and
`run_metadata.yaml`. Logs contain original-scale interval endpoints and residual
offsets, exact target indices, coverage, width, and Winkler scores. Summaries
weight each series equally, including unequal-length series. Run metadata
records package versions, worker counts, and elapsed time; per-series metadata
records split boundaries, model/cache diagnostics, and training/replay/test
timings. Enabling plotting writes the standard interval PDFs under `plots/`.
