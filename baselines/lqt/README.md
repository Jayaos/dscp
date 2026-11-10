# LQT (Areces et al., ICML 2025)

This baseline implements Linear Quantile Tracking (LQT) from
[Online Conformal Prediction via Online Optimization](https://proceedings.mlr.press/v267/areces25a.html).
The reference is the authors' experiment implementation. The method consumes
the repository's saved point forecasts; the base predictor is already trained
or pretrained.

Run from the repository root:

```bash
python sbatch/sbatch_run_lqt/run_lqt.py --dataset air --base-predictor lr
python sbatch/sbatch_run_lqt/run_lqt.py --dataset solar --base-predictor chronos --dry-run
sbatch sbatch/sbatch_run_lqt/run_lqt_air.sbatch
```

There are nine presets under [configs/lqt_configs](../../configs/lqt_configs):
Air, Solar, and Sapflux with LR, LSTM, and Chronos forecasts. Slurm launchers
are under [sbatch/sbatch_run_lqt](../../sbatch/sbatch_run_lqt). Each dataset's
array has tasks 0/1/2 for LR/LSTM/Chronos respectively. The environment overrides
are `LQT_CONDA_MODULE` (default `anaconda3`) and `LQT_CONDA_ENV` (default `dscp`).
Use `--num-cores 4` to parallelize independent sequences. Within each sequence,
predictions and updates remain chronological.

Slurm launchers pass `SLURM_CPUS_PER_TASK` to `--num-cores`, overriding the YAML
worker count. For example, `sbatch --cpus-per-task=8
sbatch/sbatch_run_lqt/run_lqt_air.sbatch` runs up to eight independent sequences
concurrently per array task during both tuning and testing. Each worker uses
one numerical-library thread. The worker count is capped by the number of
sequences; memory is configured separately through Slurm's `--mem` setting.

Use the shared `dscp` environment from
[envs/env-dscp.yml](../../envs/env-dscp.yml). LQT uses NumPy and the existing
configuration and reporting dependencies; no separate environment is needed.

## Configurable settings

This example shows the default search and the fixed settings used when tuning
is disabled:

```yaml
num_cores: 1
data:
  data_path: ./data/air-10_prediction/lr/lr_air-10_data.pkl
  calibration_ratio: 0.66
  test_ratio: 0.34

model:
  method: lqt
  target_quantiles: [[0.05, 0.95]]
  prediction_step: 1
  score_type: absolute
  learning_rate: 0.1
  learning_rate_schedule: fixed
  decay_exponent: 0.6
  p_order: 1
  bias: 1.0
  initial_quantile: 1.0

tuning:
  enabled: true
  learning_rate_grid: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
  p_order_grid: [0, 1, 2]
  bias_grid: [0.1, 1.0, 5.0, 10.0, 200.0, 100.0, 1000.0]
  coverage_tolerance: 0.01

saving_dir: ./results/lqt/example/
```

The original experiments selected hyperparameters by grid search. Accordingly,
**tuning is enabled by default**. The fixed `learning_rate: 0.1`, `p_order: 1`,
and `bias: 1.0` are explicit fallback settings for runs with tuning disabled,
not universal values selected by the authors.

- `learning_rate_schedule: fixed` uses the chosen learning rate at every update.
  `decaying` uses `learning_rate * (t + 1)^(-decay_exponent)`, with zero-based
  sequence time `t`. Both schedules were evaluated in the paper; its decay
  exponent is 0.6. The schedule is configured rather than searched.
- `p_order` determines the number of previous scores used as lag features.
  `bias` scales the constant feature.
- LQT initializes all coefficients at `1 / p_order` when `p_order > 0`.
  When `p_order == 0`, its single coefficient is `initial_quantile`.
- `score_type: absolute` uses `abs(y - point_prediction)` in original response
  units. No score normalization, threshold clipping, or coefficient projection
  is applied.
- Symmetric `target_quantiles` pairs specify coverage; `[0.05, 0.95]` requests
  90%. Each sequence and requested coverage level has its own selected
  hyperparameters and online state.

For the exact source, initialization details, and documented differences from
the generic `conformalopt` package, see [UPSTREAM.md](UPSTREAM.md).

## Data split, tuning, and test state

For a saved sequence of length T, testing starts at
`floor(T * calibration_ratio)`, using NexCP's floating-point arithmetic.
The prefix is calibration and the entire suffix is test. An optional
`data.test_start` integer, or mapping from sequence keys to integers, can
override the boundary. Both partitions must be nonempty. The artifact needs
scalar `heldout_y` and `heldout_predictions` arrays; covariates are not needed.

The whole calibration prefix is used for chronological candidate evaluation;
there is no further calibration/validation split. Each candidate starts with
fresh state and predicts before observing each calibration outcome. Selection
is independent for each sequence and coverage level:

1. Compute mean pinball loss at quantile `1 - nominal_alpha` and coverage
   over the entire calibration prefix.
2. Among candidates whose absolute coverage gap is at most
   `coverage_tolerance`, choose the lowest loss.
3. If no candidate meets that condition, choose the lowest loss over the
   full grid. Equal losses retain the first candidate in grid order.

This reproduces the experimental code's two-sided coverage condition. The
paper's prose describes a one-sided lower coverage bound; the distinction is
recorded in [UPSTREAM.md](UPSTREAM.md). Test outcomes never select
hyperparameters.

After selection, testing starts with **fresh coefficients, score history,
and learning-rate clock**. Calibration is used for selecting
hyperparameters, not for seeding test history. This follows the authors'
experiment runner. With tuning disabled, the configured fixed parameters are
used with the same fresh test state.

At each test timestamp, the method predicts a score threshold, constructs its
prediction set around the saved point forecast, observes the target, and then
updates its state. Equality with a threshold is covered. Evaluation uses
one-step sequential feedback, including for the existing Chronos forecasts.

The default grid contains 231 LQT candidates per sequence and coverage level.

## Prediction sets and metrics

For a nonnegative threshold `q`, the interval is
`[point_prediction - q, point_prediction + q]`. A zero threshold is a singleton
interval and remains in finite-interval metrics. A negative threshold gives an
empty set; it is not clipped to zero. The first `p_order` thresholds are zero
under the authors' initialization.

The reporting convention matches the ACI baseline:

- Coverage uses all test timestamps, including empty or full sets.
- Mean interval width and mean Winkler score use finite, nonempty intervals.
  Undefined conditional means are stored as null.
- Full/empty counts, rates, masks, and full-test diagnostic means are recorded.
  The diagnostic convention assigns infinite score to full and empty sets;
  empty-set width is zero and full-set width is infinite.
- Ordinary Winkler penalties use the fixed nominal miscoverage level.
  Sequence summaries give each sequence equal weight; pooled exception rates
  are also available.

The algorithm has no intentional full-set mechanism. Nonfinite arithmetic
is an error, not an instruction to replace predictions
with a full set. During tuning, failed candidates are recorded and excluded
from selection; if every candidate fails, the run stops. Test-time failures
always stop the run.

## Outputs and replay

Runs write `log.pkl`, `summary_results.pkl`, `requested_config.yaml`, and
`resolved_config.yaml`. Runs using a saved or tuned selection also write
`selected_hyperparameters.pkl`; tuning additionally writes `tuning_results.pkl`. Optional plots use the original target indices.
The selected-parameter artifact records settings separately for every sequence
and coverage pair.

The resolved configuration disables tuning and uses
`selected_hyperparameters_path` to replay the selected settings. Keep that
artifact with the resolved configuration. A replay restarts the online test
state; it does not resume a partially completed stream.

The shared result inspector understands the conditional width/score convention:

```bash
python -m report.inspect_run_results results/lqt/example/
```
