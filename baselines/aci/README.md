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

Normal runs use the fixed values in `model`. Hyperparameter selection is a
separate step using the dedicated tuning runner below. Normal configs have no
`tuning` section; stale tuning sections are rejected with guidance to the
dedicated runner. Remove that section when migrating an older run config.

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

## DSCP-QR-style hyperparameter selection

The dedicated tuning runner uses the same selection procedure as DSCP-QR:

1. Evaluate each gamma/window candidate on the same selected sequences and
   configured coverage levels, using fresh independent ACI states.
2. Require `observed_coverage - target_coverage > tuning.delta_threshold`
   for **every** sequence/coverage pair. The default threshold is -0.01.
3. Rank passing candidates by mean Winkler score, weighting sequences and
   coverage levels equally. Ties follow grid order. Save the best `top_k` trials.

ACI's previously agreed reporting convention still applies: coverage includes
full/empty sets, while Winkler and width exclude them. Their counts/rates are
reported separately and do not receive a ranking penalty. A candidate with
undefined finite-interval Winkler at any sequence/level cannot be selected.
If no candidate qualifies, the result has no top trials and no fallback choice.

This selects **one shared gamma/window pair across the selected sequences and
coverage levels**, not separately for each sequence. Independent online states
are created for every sequence/level. This search protocol and its grid are
repository benchmarking choices, not the original ACI paper's tuning procedure.
The ordinary ACI presets use fixed gamma 0.005 and window 1250, with no
`tuning` section.

### Dedicated tuning and separate final evaluation

```bash
python -m sbatch.sbatch_run_tuning.run_aci_tuning \
  --base-config configs/aci_configs/aci_lr_air_config.yaml \
  --grid-config configs/aci_configs/aci_air_tuning_config.yaml \
  --save-dir results/tuning/aci_lr_air --top-k 3

python -m report.inspect_tuning_results \
  results/tuning/aci_lr_air/tuning_results.pkl --top-k 3
```

The three `configs/aci_configs/aci_{air,solar,sapflux}_tuning_config.yaml` grids
search gamma `[0.001, 0.005, 0.01, 0.05]` and window size `[100, 250, 500, 1250]`.
They select 3 Air, 10 Solar, or 5 Sapflux sequences, matching QR's grids. Keys
are sorted; `--sequence-index` chooses the start of the contiguous subset.
`tuning.num_sequences: all` selects every sequence, and `--sequence-key KEY`
overrides the subset with one sequence. All settings are configurable in YAML.

The grids set `tuning.initialization_ratio: 0.5`, a fraction of the **whole saved
sequence**, to match QR's outer validation region approximately: first 50%
initialize the score history, next 16% evaluate candidates, final 34% remain
untouched. The test boundary uses NexCP's exact floor rule, so rounding can
change a boundary by one observation relative to QR. ACI has no neural training
or checkpoint-selection stage. For 10,000 observations the ranges are
`[0,5000)` history, `[5000,6600)` tuning evaluation, `[6600,10000)` final test.

The runner writes `tuning_results.pkl` plus each trial's `result.pkl` and
`resolved_config.yaml`. It never evaluates the final test set. The resolved
config contains fixed model settings and no `tuning` section, and can be
passed directly to the ordinary ACI runner. Use the best trial path printed by
the tuner or inspector, for example if trial 7 wins:

```bash
python -m sbatch.sbatch_run_aci.run_aci \
  results/tuning/aci_lr_air/trial_0007/resolved_config.yaml
```

Final evaluation applies the shared selected settings to **all** sequences in
the artifact. It recreates each state from the full 66% calibration prefix and
resets alpha to its specified initial value. No validation state carries over.
Slurm tuning arrays for all nine dataset/predictor combinations are documented
in [the job guide](../../sbatch/sbatch_run_aci/README.md#tuning-jobs).
`--dry-run` validates the dedicated runner's config without loading data or
writing files. `--seed` is recorded for consistency with QR; ACI is deterministic.

The dedicated grid also accepts `tuning.validation_ratio` when
`initialization_ratio` is omitted or null. It controls the fraction of
calibration reserved for candidate evaluation: the default 0.2 gives 5,280
initialization, 1,320 validation, and 3,400 final-test points in a sequence of
10,000 observations. The supplied grids' `initialization_ratio: 0.5` takes
precedence and gives the 50%/16%/34% split above. These selection settings
belong only to the dedicated tuning workflow.

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
`resolved_config.yaml`, and optionally `plots/`. The resolved configuration
contains the fixed model settings and no `tuning` section, so it can be replayed
directly. The separate tuning runner writes `tuning_results.pkl` and trial
artifacts as described above. Per-point logs retain alpha before/after update,
thresholds, scores, endpoints, coverage, and indices.

The shared inspector recognizes the declared conditional-efficiency scope:

```bash
python -m report.inspect_run_results results/aci/example/
```
