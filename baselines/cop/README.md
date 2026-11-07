# COP (Hu et al., ICLR 2026)

Conformal Optimistic Prediction (COP) combines an online primary threshold
update with a correction based on the empirical distribution of past scores.
This baseline follows the authors' published implementation of
[Distribution-informed Online Conformal Prediction](https://arxiv.org/html/2512.07770v2)
and consumes the repository's saved point forecasts.

Run from the repository root:

```bash
python sbatch/sbatch_run_cop/run_cop.py --dataset air --base-predictor lr
python sbatch/sbatch_run_cop/run_cop.py --dataset solar --base-predictor chronos --dry-run
```

Nine presets are provided under
[configs/cop_configs](../../configs/cop_configs), covering Air, Solar, and Sapflux
with LR, LSTM, and Chronos forecasts. The corresponding Slurm arrays are under
[sbatch/sbatch_run_cop](../../sbatch/sbatch_run_cop). Use `--num-cores 4` to
parallelize independent sequences. Within each sequence, prediction and feedback
remain chronological.

COP runs in the existing [dscp environment](../../envs/env-dscp.yml). Its active
ECDF implementation needs NumPy; the runner uses dependencies already present
in that environment. No additional package or separate environment is needed.

## Configuration

The following shows the default algorithm settings and hyperparameter search:

```yaml
num_cores: 1

data:
  data_path: ./data/air-10_prediction/lr/lr_air-10_data.pkl
  calibration_ratio: 0.66
  test_ratio: 0.34

model:
  target_quantiles: [[0.05, 0.95]]
  learning_rate: 0.1
  window_size: 100
  scale: 0.5
  proportional_lr: true
  initial_quantile: 0.0
  cdf: ecdf
  score_type: signed
  initialization: calibration_replay
  prediction_step: 1

tuning:
  enabled: true
  learning_rate_grid: [1.0, 0.5, 0.1, 0.05]
  window_size_grid: [100]
  scale_grid: [0.5]
  coverage_tolerance: 0.01
  warmup_steps: null

plotting:
  plotting: true
  plotting_seq_len: 200

saving_dir: ./results/cop/example/
```

Tuning is enabled by default. The four learning-rate candidates follow the
authors' experiment configurations. The fixed `learning_rate: 0.1` is a
fallback for runs with tuning disabled, not a universally selected publication
value. The window and scale defaults follow the paper's general setup; their
grids can be expanded explicitly.

- `window_size` controls the past-score range used by the adaptive learning
  rate, the ECDF window, and the number of initial steps without ECDF correction.
  The default is 100. The authors used dataset-specific windows of 300 and 500
  in their electricity and climate configurations.
- `scale` multiplies the distribution-informed correction relative to the
  learning rate. Its publication default is 0.5.
- With `proportional_lr: true`, the effective learning rate at time `t > 0`
  is `learning_rate * (max(past_scores) - min(past_scores))`, using at most
  `window_size` scores strictly before `t`. At time zero it is the base rate.
  A constant past window therefore gives a zero effective rate. With
  `proportional_lr: false`, the base rate stays constant.
- `initial_quantile` initializes the primary and issued thresholds of a fresh
  COP stream; the authors initialize both to zero.
- `cdf: ecdf` is the published default. This implementation does not replace
  it with a learned distribution model or the paper's optional KDE experiment.
- `prediction_step: 1` uses sequential feedback, including with the existing
  Chronos forecast artifacts.
- `target_quantiles: [[0.05, 0.95]]` requests 90% nominal interval coverage.

See [UPSTREAM.md](UPSTREAM.md) for the exact source, update indexing, and
distinctions between the publication code and this benchmark integration.

## Scores and prediction sets

The default `score_type: signed` follows the authors' real-data experiments.
For each target interval, two independent COP streams use:

- Lower score: `point_prediction - y`.
- Upper score: `y - point_prediction`.

Each stream uses half of the nominal interval miscoverage. If their thresholds
are `q_lower` and `q_upper`, the resulting interval is
`[point_prediction - q_lower, point_prediction + q_upper]`.
The two thresholds can differ, so the interval can be asymmetric around the
point forecast.

A negative individual signed threshold is valid and does not by itself imply
an empty set. Only reversed final endpoints define an empty interval. A zero
width interval is a singleton and remains in finite, nonempty metrics.

The optional `score_type: absolute` instead uses one COP stream on
`abs(y - point_prediction)` at the full nominal miscoverage and constructs
`[point_prediction - q, point_prediction + q]`. A negative threshold in this
mode does define an empty set. This is a supported score choice, but it differs
from the authors' default real-data configuration.

Scores and endpoints remain in the saved forecasts' original response units.
There is no residual normalization or new base-predictor training. In particular,
COP's default signed scores differ from the absolute-score defaults of the ACI
and LQT/SQT+AR baselines; that distinction is saved in the configuration.

## Split, tuning, and online state

Testing begins at `floor(T * calibration_ratio)` for each saved sequence,
matching NexCP's floating-point split calculation. The prefix is calibration
and the entire suffix is test. An optional `data.test_start` integer or
per-sequence mapping can override this boundary. Both partitions must be
nonempty. Artifacts need scalar `heldout_y` and `heldout_predictions`; no
covariates are required.

Hyperparameters are selected separately for each sequence and requested
coverage level. With signed scores, one candidate supplies the same learning
rate, window, and scale to both tails; the tails have independent online states.
Each candidate runs chronologically from fresh state on calibration only.

For selection, `warmup_steps: null` excludes the first
`max(window_size_grid) + 1` calibration predictions from ranking. Every
candidate is compared on the same timestamps. With the default single window
of 100, this follows the authors' plotting/evaluation cutoff of 101. A specified
integer overrides that common cutoff.
Excluded startup observations still update the candidate state; they are not
removed from its history. Calibration must contain observations after the
configured cutoff.

The selection rule is a benchmark protocol, because the authors' runner
evaluates every rate and does not provide an automatic calibration selector:

1. Compute interval coverage and mean score-space pinball loss on the retained
   calibration predictions. For signed scores, add the two tail losses.
2. Among candidates with interval coverage at least
   `nominal_coverage - coverage_tolerance`, select the lowest loss.
3. If none meets that condition, select the lowest loss over the candidate grid.

The coverage condition is applied to the complete two-sided interval, not to
either tail separately. Selected parameters remain fixed throughout testing;
test outcomes only update online state.

With the default `initialization: calibration_replay`, the selected method is
restarted and replayed through calibration, then continues into test without a
state reset. This preserves the uninterrupted stream used by the authors while
introducing the requested calibration/test boundary. Each sequence, coverage
level, and signed tail retains separate state.

The optional `initialization: cold_start` starts a fresh COP stream at the test
boundary. It is an explicit alternative protocol, not the default. All test
timestamps remain in evaluation, including any startup predictions; no test
warmup is silently discarded. COP's default calibration replay differs from
the fresh-test-state protocol of the LQT/SQT+AR experiment implementation.

## Reporting and outputs

Prediction sets follow their mathematical endpoints. The upstream experiment
wrapper sorts crossed endpoints; this baseline preserves them as empty sets,
consistent with the repository's agreed exceptional-set convention.

- Coverage includes every test timestamp, including empty/full sets.
- Width and Winkler means are conditional on finite, nonempty intervals.
  Undefined conditional means are stored as null.
- Full/empty counts, rates, masks, and full-test diagnostics are saved
  separately. Empty-set diagnostic width is zero; full-set width is infinite.
  The diagnostic score convention assigns infinity to full and empty sets.
- Ordinary Winkler penalties use the fixed nominal interval miscoverage.
  Sequence metric summaries give each sequence equal weight, while pooled
  exceptional-set rates are also available.

The active finite-score COP recurrence has no intentional full-set mechanism.
Numerical failures are surfaced rather than converted into full prediction
sets. No per-time-step coverage guarantee is claimed by this implementation.

Runs write `requested_config.yaml`, `resolved_config.yaml`, `log.pkl`, and
`summary_results.pkl`. Runs using selected parameters also write
`selected_hyperparameters.pkl`; tuning additionally writes
`tuning_results.pkl`. Optional plots use the original target indices.

The resolved configuration disables tuning and points to the selected-parameter
artifact through `selected_hyperparameters_path`. Keep that artifact with the
resolved configuration to replay the same settings. Replay recomputes the
configured calibration/test state protocol; it does not resume a partially
completed stream.

Inspect results with:

```bash
python -m report.inspect_run_results results/cop/example/
```
