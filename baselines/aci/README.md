# Adaptive conformal inference (ACI)

This baseline implements the original simple update of Gibbs and Candes,
[Adaptive Conformal Inference Under Distribution Shift (2021)](https://arxiv.org/pdf/2106.00170),
using saved point forecasts. It needs calibration and test partitions only.
The forecasting model has already been fitted (or is pretrained); ACI does not
train a separate offline model.

Run from the repository root:

```bash
python -m sbatch.sbatch_run_aci.run_aci --dataset air --base-predictor lr
python -m sbatch.sbatch_run_aci.run_aci configs/aci_configs/aci_lr_air_config.yaml --dry-run
```

There are nine presets for Air, Solar, and Sapflux with LR, LSTM, and Chronos.
Slurm instructions are in [the job guide](../../sbatch/sbatch_run_aci/README.md).

## Defaults and configurable settings

```yaml
num_cores: 1
data:
  data_path: ./data/air-10_prediction/lr/lr_air-10_data.pkl
  calibration_ratio: 0.66
  test_ratio: 0.34

model:
  target_quantiles: [[0.05, 0.95]]
  gamma: 0.005
  window_size: 1250
  initial_alpha: null
  quantile_method: linear
  score_type: absolute
  prediction_step: 1

tuning:
  enabled: false
  validation_ratio: 0.2
  gamma_grid: [0.001, 0.005, 0.01, 0.05]
  window_size_grid: [1250]
  coverage_tolerance: 0.01

plotting:
  plotting: true
  plotting_seq_len: 200
saving_dir: ./results/aci/example/
```

- `target_quantiles` encodes nominal coverage. Pairs must be symmetric, but
  either order is accepted; [0.05, 0.95] requests 90% coverage. Multiple pairs
  create independent ACI states.
- `gamma` is the constant adaptation step size. The default 0.005 follows the
  publication. Zero is available as a fixed-level control.
- `window_size` caps the most recent observed calibration scores. The default
  1250 follows the paper's volatility experiment; null uses expanding history.
  A shorter calibration prefix uses all available scores.
- `initial_alpha: null` initializes each state at its nominal miscoverage.
  A number in [0, 1] overrides this initial value. There is no subsequent
  clipping or projection of the adaptive level.
- `quantile_method: linear` matches the authors' R implementation (type 7).
  `inverted_cdf` instead follows the empirical-CDF inverse displayed in the
  paper. The selected method is saved in the run metadata.
- `score_type: absolute` uses |y - prediction| in original response units.
  This is the paper's generic point-prediction score. `relative` uses
  |y - prediction| / prediction, as in its positive variance-forecast example,
  and requires strictly positive forecasts. Its interval radius is rescaled by
  the current point prediction. There is no automatic centering, normalization,
  clipping, or small-denominator replacement.
- `num_cores` parallelizes independent sequences. Predictions within each
  sequence always remain chronological. The implementation is CPU-only.

See [UPSTREAM.md](UPSTREAM.md) for the exact source and fidelity decisions.

## Splits and causal initialization

For a saved sequence of length T, the default test boundary is
`floor(T * calibration_ratio)`, using exactly NexCP's floating-point arithmetic.
The earlier prefix is calibration; every remaining observation is a test target.
An optional `data.test_start` integer, or mapping from series keys to integers,
overrides the boundary for comparisons on identical target indices. Ratios are
still validated, and both partitions must be nonempty. Covariates are optional;
only scalar `heldout_y` and `heldout_predictions` arrays are required.

Calibration seeds the rolling score pool. The adaptive level starts fresh at
the beginning of testing; calibration outcomes do not first update that level.
At each test timestamp, ACI issues its interval, observes the outcome, updates
alpha by `gamma * (nominal_alpha - miss_indicator)`, then appends the score.
Equality with an endpoint is covered. Each series and coverage level has its
own independent state.

Evaluation uses one-step sequential feedback. Existing Chronos forecasts may
have been generated in blocks: their intervals are recalibrated as outcomes
arrive. The same saved forecasts should be used by all compared methods.

## Optional calibration-only hyperparameter selection

Set `tuning.enabled: true` to search the configured gamma/window grid.
Hyperparameters are shared across the artifact's sequences and coverage levels;
online states remain independent. The final test suffix never participates.

The early `1 - validation_ratio` fraction of calibration initializes each
candidate. The remaining calibration tail evaluates it sequentially with fresh
states. For 10,000 saved forecasts, defaults give 5,280 initialization points,
1,320 tuning-validation points, and 3,400 final-test points.

A candidate passes coverage if every series/coverage pair reaches its nominal
coverage minus `coverage_tolerance` and has a defined finite-interval mean score.
Candidates are ranked lexicographically by:

1. Passing the coverage/defined-score condition.
2. Maximum coverage shortfall.
3. Mean full/empty-set frequency.
4. Mean Winkler score among finite, nonempty intervals.

Series/coverage pairs have equal weight. Ties follow grid order. This selection
rule is our benchmark protocol, not a claim about the original authors' tuning.
After selection, each final state is recreated using the entire calibration
prefix to seed its score pool, and alpha is reset to its specified initial value.

## Full/empty sets and output metrics

Original ACI returns a full set when its adaptive alpha is <= 0, and an empty
set when alpha is >= 1. These cases remain in the timeline and in adaptive
updates. They are never converted to finite intervals or dropped from coverage.

- Coverage and coverage gap use **all test timestamps**.
- `avg_interval_width` and `avg_winkler_score` summarize **finite, nonempty
  intervals only**. They are null when no such intervals exist.
- Counts/rates of full and empty sets, finite masks, and full-test diagnostic
  means are also saved. Full width is infinity; empty-set width is zero.
- The extended full-test score assigns infinity to full and empty sets. For an
  empty set this is an explicit diagnostic convention, not the ordinary
  ordered-interval Winkler formula.
- Winkler penalties always use the fixed nominal alpha, never adaptive alpha.
  Summary means weight sequences equally and report how many sequences have
  undefined conditional efficiency. Exception counts also provide pooled rates.
- Plots retain the true target indices, shade finite intervals, and mark full
  and empty predictions separately.

Outputs: `log.pkl`, `summary_results.pkl`, `requested_config.yaml`,
`resolved_config.yaml`, and optionally `plots/`. Tuning also writes
`tuning_results.pkl`. The resolved configuration contains the selected settings
and disables tuning, so it can be replayed directly. Per-point logs retain
alpha before/after update, thresholds, scores, endpoints, coverage, and indices.

The shared inspector recognizes the declared conditional-efficiency scope:

```bash
python -m report.inspect_run_results results/aci/example/
```
