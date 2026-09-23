# DSCP

## SplitCP baseline

SplitCP calibrates a fixed absolute-residual interval independently for each
series using saved base forecasts. It follows the unweighted, unscaled
procedure in Tibshirani's `conformalInference/R/split.R`, including the
finite-sample quantile correction and infinite-radius small-sample case.
Only calibration and test partitions are needed; there is no CP training or
validation stage and no test-time calibration update.

```bash
python -m sbatch.sbatch_run_split_cp.run_split_cp --dataset air --base-predictor lr
python -m sbatch.sbatch_run_split_cp.run_split_cp --dataset solar --base-predictor lstm --dry-run
```

Nine presets use 66% calibration / 34% test. Ratios are configurable, and exact
per-series test-start indices can align comparisons despite differences in
other methods' split rounding. The runner writes the existing log, summary,
resolved-configuration, and plotting formats. See the
[SplitCP guide](baselines/split_cp/README.md) and
[reference notes](baselines/split_cp/UPSTREAM.md) for details.

## DistMatch baseline

DistMatch reads saved base-predictor forecasts and estimates intervals from
signed residual windows. Its experiment YAML controls all three chronological
partitions and the number of parallel sequence workers:

```yaml
num_cores: 4
threads_per_worker: 1
data:
  train_ratio: 0.50
  valid_ratio: 0.16
  test_ratio: 0.34
```

The ratios partition each complete saved held-out suffix and must sum to one.
Training and test must be positive; `valid_ratio: 0` is allowed for fixed
settings without tuning. The matching trees fit on training only. Validation
observations then enter their leaves in chronological order before final-test
evaluation. Each new test residual becomes available only after its interval
is issued. There is no additional calibration partition.

All LR, LSTM, and Chronos presets enable `data.normalize_residual: true`, which
is also the default when omitted. Target statistics use the held-out prefix
observed before the evaluated split, with saved `train_y` prepended when
available. Chronos artifacts without `train_y` use that held-out prefix alone.
Both residual windows and quantile-forest targets are divided by the frozen
target sample standard deviation; quantiles are restored to original units for
intervals and metrics. Set `data.normalize_residual: false` to use raw residuals.
Saved point forecasts are reused without retraining the forecasting models.
The former `data.normalize` and `data.normalization_mode` keys are rejected;
the guide documents migration and the exact statistics.

From the repository root, after activating the environment defined in
[`envs/env-distmatch.yml`](envs/env-distmatch.yml):

```bash
python -m sbatch.sbatch_run_distmatch.run_distmatch configs/distmatch_configs/distmatch_lr_air_config.yaml
```

`num_cores` is the maximum number of independent series evaluated concurrently;
timestamps within each series remain sequential. Omitting `--num-cores` uses
the YAML value. All workers have reproducible series-specific random streams.

See the [DistMatch guide](baselines/distmatch/README.md) for presets, validation-only
tuning, cache controls, and batch execution, and the
[provenance notes](baselines/distmatch/UPSTREAM.md) for reference compatibility.

## ResCP

The ResCP baseline applies a fixed random reservoir to signed point-forecast
residuals. It samples previously observed residuals according to reservoir-state
similarity and builds prediction intervals from the sampled quantiles. There is
no CP neural-network training stage. The implementation follows the official
sampling code at commit `1d8e560`; see
[`baselines/rescp/UPSTREAM.md`](baselines/rescp/UPSTREAM.md) for the exact
recurrence, recency weighting, numerical choices, and documented corrections.
The upstream MIT notice is retained alongside the adapter.

### Calibration/test sizes and tuning validation

Ordinary runs split each saved base-predictor held-out sequence into calibration
and test using two configuration settings:

```yaml
data:
  data_path: ./data/air-10_prediction/lstm/lstm_air-10_data.pkl
  calibration_ratio: 0.66
  test_ratio: 0.34
  normalize: true
```

Both ratios must be finite, positive, and sum to one. For a sequence of length
`T`, the nominal calibration size is `C = floor(T * calibration_ratio)` and
test contains the remaining `T - C` observations. Floating-point values close
to integer boundaries are rounded before the outer floor. Empty partitions
are rejected. No strided windows are constructed, so every evaluation target
is scored.

The Sapflux tuning YAML reserves a tail of that nominal calibration prefix
for hyperparameter selection:

```yaml
tuning:
  model_selection_valid_ratio: 0.2
```

This ratio is relative to the calibration prefix, not the full held-out
sequence. It must be finite and strictly between zero and one; it defaults to
`0.2`. Tuning initializes on
`floor(nextafter(C * (1 - model_selection_valid_ratio), +inf))` observations
and evaluates the rest of calibration. `nextafter` nudges a floating-point
product toward the next representable value to stabilize integer boundaries.
Both inner regions must be nonempty. The final test suffix is excluded from
tuning.

| Run stage | Use |
| --- | --- |
| Tuning initialization | Fit optional residual-input normalization on the earlier calibration prefix and initialize reservoir state and residual memory. |
| Tuning validation | Evaluate candidates sequentially on the remaining calibration tail. Earlier revealed validation residuals may enter later intervals. |
| Final test | Initialize a fresh estimator and fit normalization on the full nominal calibration prefix, then evaluate the test suffix with the selected fixed settings. The tuning ratio is ignored. |

For `T = 10000` with `0.66 / 0.34` and a tuning ratio of `0.2`, tuning uses
5280 initialization observations and 1320 validation targets, reserving 3400
test observations. The final run uses all 6600 calibration observations and
evaluates those same 3400 test targets.

`data.calibration_ratio` controls the nominal calibration partition.
`model.calibration_size` independently caps the number of state/residual pairs
kept in rolling memory; set it to `null` for expanding memory. With
`sampling_num: null`, the fixed Monte Carlo sample count is the memory cap,
or the initial calibration length when memory is uncapped.

The default `0.66 / 0.34` split reserves approximately the same test suffix as
ordinary QR-CP/SPCI/KOWCPI. Use `0.80 / 0.20` for the IQN/Local-CP final suffix.
Compare the saved exact `target_indices` within a common artifact, since other
baselines can round boundaries differently. For example, at `T = 101`, the
new default starts test at index 66; the previous `0.50 / 0.16 / 0.34` split
started at 67. All split settings, including the tuning validation ratio,
stay fixed outside the candidate grid so candidates evaluate identical targets.

To migrate an old ResCP config, combine its calibration and validation ratios
into `data.calibration_ratio`, retain `data.test_ratio`, and remove
`data.validation_ratio`. The old validation key is rejected. Configure tuning
validation separately with `tuning.model_selection_valid_ratio` if needed.

### Online data use and method settings

At timestamp `t`, the query state contains residuals only through `t-1`. Each
stored residual `r_j` is paired with the state before `r_j` was observed.
ResCP emits the interval before adding `r_t` and advancing the reservoir.
State advances through initialization and evaluation within each run and is
independent for each series.
The adapter accepts scalar targets/predictions in `[T]` or `[T, 1]` format and
supports unequal sequence lengths. It does not require `heldout_x`.

Normalization affects reservoir inputs only. The sampling pool, logged
residual quantiles, final interval endpoints, widths, and Winkler scores remain
in original units. Normalization is fitted once per run: on the earlier
initialization prefix during tuning, or on the full calibration prefix for
final test. It remains frozen throughout that run's evaluation.

All presets use reservoir size 512 and connectivity 0.2. Solar and Air use
the fixed selected ResCP hyperparameters from
[Table 4 of the paper](https://arxiv.org/pdf/2510.05060#page=20), with the
experiment mapping RNN -> DSCP LSTM, Transformer -> Chronos, and ARIMA -> LR.
Air uses the paper's Beijing settings.

| Dataset | DSCP predictor | Paper predictor | Spectral radius | Leak rate | Input scaling | Temperature | Memory cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Solar | LSTM (RNN) | RNN | 0.9 | 0.75 | 0.7 | 0.1 | 3900 |
| Solar | Chronos | Transformer | 0.9 | 0.8 | 0.4 | 0.1 | 7600 |
| Solar | LR | ARIMA | 1.0 | 0.8 | 0.2 | 0.1 | 1500 |
| Air | LSTM (RNN) | RNN | 1.3 | 0.95 | 0.25 | 0.1 | 3200 |
| Air | Chronos | Transformer | 1.0 | 0.8 | 0.25 | 0.1 | 10500 |
| Air | LR | ARIMA | 1.45 | 0.65 | 0.75 | 0.15 | 3800 |

Solar and Air runs use these presets directly without hyperparameter tuning,
initializing on the full calibration prefix. These are transfers of the
published ResCP settings to DSCP's base forecasts; the base predictors and data
partitions follow the DSCP experiment.

Only Sapflux is tuned, separately for each base predictor. Its starting
settings are spectral radius 1.2, leak 0.9, input scaling 0.25, temperature 0.1,
and memory cap 3800. The Sapflux tuning grid searches 576 joint combinations
of spectral radius, leak rate, input scaling, temperature, and memory cap,
using five sequences by default (`tuning.num_sequences: 5`).

`recurrence: upstream` preserves the executable upstream update, whose tanh
term has no additional leak multiplier. `decay: linear` uses upstream's
oldest-first ramp `[0, ..., n-1]`; `none` and `exponential` are also supported.
These recurrence/ramp definitions differ from the corresponding paper
equations and are documented in the provenance file.

`use_beta_search: true` selects the narrowest interval among `beta_bins: 100`
candidate lower-tail probabilities. Standard Winkler scoring then uses the
nominal miscoverage alpha. With beta search disabled, the exact
`target_quantiles` pairs are used with DSCP's tail-specific scoring.
The adapter enforces the configured memory cap and fixed sampling count,
handles zero states/singleton histories, and uses isolated seeded random
streams. Residual clipping and adaptive-alpha updates are disabled.

Only `model.prediction_step: 1` is supported. Chronos artifacts with flattened
ten-step blocks can be recalibrated only under the sequential-observation
protocol described below; use newly generated one-step artifacts for a
homogeneous one-step forecasting comparison.

### Run and tune ResCP

Create and activate the dedicated CPU environment from the repository root:

```bash
conda env create -f envs/env-rescp.yml
conda activate rescp
```

[`envs/env-rescp.yml`](envs/env-rescp.yml) contains the dependencies for running
and tuning ResCP on saved forecasts, including interval plots. Generate base
predictor artifacts using their existing environments. The ResCP Slurm scripts
activate `rescp` by default; use `--export=ALL,RESCP_ENV=dscp` to select the
existing general environment instead.

Then run an experiment:

```bash
python -m sbatch.sbatch_run_rescp.run_rescp configs/rescp_configs/rescp_lstm_air_config.yaml
python -m sbatch.sbatch_run_rescp.run_rescp --dataset solar --base-predictor chronos --dry-run
python -m sbatch.sbatch_run_rescp.run_rescp --dataset sapflux --base-predictor lstm --num-cores 4
```

Nine configuration presets cover Air, Solar, and Sapflux with LR, LSTM, and
Chronos artifacts. The corresponding saved base forecasts must already exist.
CPU Slurm array scripts are in `sbatch/sbatch_run_rescp/`; each dataset array
runs its three base predictors. CLI `--seed` and `--output-dir` overrides
support separate experiment runs. `threads_per_worker` defaults to one to
avoid oversubscribing CPUs when processing independent series in parallel.

For Sapflux, tune on the reserved calibration tail and then run the exported
`best_config.yaml` directly for final test:

```bash
python -m sbatch.sbatch_run_tuning.run_rescp_tuning \
  --base-config configs/rescp_configs/rescp_lstm_sapflux_config.yaml \
  --grid-config configs/rescp_configs/rescp_sapflux_tuning_config.yaml \
  --save-dir results/rescp_tuning/sapflux_lstm

python -m sbatch.sbatch_run_rescp.run_rescp results/rescp_tuning/sapflux_lstm/best_config.yaml
```

Repeat with the LR and Chronos Sapflux presets, using separate output folders
`results/rescp_tuning/sapflux_lr` and `results/rescp_tuning/sapflux_chronos`.

To tune all three Sapflux base predictors on Slurm, submit from the repository root:

```bash
sbatch sbatch/sbatch_run_tuning/run_rescp_sapflux_tuning.sbatch
```

Array tasks 0/1/2 tune LR/LSTM/Chronos with four CPU workers per task in the
`rescp` environment. The job reads the current Sapflux tuning YAML, including
`tuning.model_selection_valid_ratio` and `tuning.num_sequences`. Results go to
`results/tuning/rescp_sapflux/job_<array-job-id>/<predictor>/`; run the exported
`best_config.yaml` there for final test. Environment overrides include
`DSCP_RUNPATH`, `RESCP_ENV`, `RESCP_SEED`, `RESCP_GRID_CONFIG`, and
`RESCP_TUNING_OUTPUT_ROOT`. Additional script arguments are forwarded to the tuner.

Tuning evaluates only validation. Candidates use the same seeds and are ranked
by mean per-series/per-interval Winkler score, optionally filtered by
`tuning.delta_threshold` for coverage gap (set it to `null` to disable the
filter). The tuner writes trial artifacts, `tuning_results.pkl`, and
`best_config.yaml` when an eligible candidate exists. It does not run test;
the second command initializes afresh on full calibration and performs final
test evaluation, ignoring the saved tuning validation ratio. If no candidate
passes the coverage threshold, the results record that outcome without
selecting one.

Ordinary runs save `resolved_config.yaml`, `log.pkl`, `summary_results.pkl`,
and optional `plots/`. Logs use the current DSCP schema: `lower_interval` and
`upper_interval` are final response-scale endpoints, with separate raw
`lower_residual_quantile` and `upper_residual_quantile` fields. They also record
selected beta, exact target indices, scaler statistics, seeds, runtime, and
upstream revision. Repeat selected configurations with different seeds when
reporting reservoir/sampling variability.

## QR-CP prediction heads

RNN and Transformer QR-CP support two choices through `model.head_type`:

| Value | Quantile prediction |
| --- | --- |
| `nondecreasing` (default) | Predicts the lowest quantile directly and adds cumulative softplus increments for higher quantiles, guaranteeing nondecreasing outputs. |
| `independent` | Each quantile head directly predicts its residual quantile, `q_tau = g_tau(h)`. Quantile crossing is possible. |

To select independent heads, change this field in your QR-CP YAML:

```yaml
model:
  head_type: independent
```

Both choices use the same shared encoder, quantile loss, and sorted set of
quantile levels. Independent predictions and interval endpoints are not sorted
after prediction. Omitting `head_type` preserves the existing nondecreasing
method. The QR-CP pipeline currently supports `model.prediction_step: 1`.

The tuning runner inherits the choice from the base configuration. To compare
both choices in a grid search, uncomment `model.head_type` in the QR-CP tuning
YAML. Use a different `saving_dir` for each ordinary run when comparing the
methods so that model checkpoints and results are retained separately.

## IQN-CP prediction heads

RNN and Transformer IQN-CP support two choices through
`model.prediction_head`:

| Value | Quantile prediction |
| --- | --- |
| `partially_monotonic` | Implements the partially monotonic head. The quantile level is supplied directly, all weights along the quantile-dependent path are positive softplus transforms, and inference evaluates `g(h, tau)` directly. Quantiles are nondecreasing in `tau` by construction. |
| `cosine_embedding` (default when the selector is omitted) | Implements the paper-style cosine quantile embedding and multiplicative context conditioning. Its raw head is not constrained to be monotonic, and `model.interval_mode` selects direct evaluation or sampling-based empirical rearrangement when constructing intervals. |

The checked-in IQN-CP experiment configurations explicitly select a prediction
head. To switch between the partially monotonic and paper-style cosine designs,
change the selector:

```yaml
model:
  prediction_head: partially_monotonic # or cosine_embedding
  interval_mode: sampling # cosine only: direct or sampling
  sampling_num: 1000 # cosine sampled validation/inference; separate from num_taus
  cos_emb_dim: 32 # cosine feature count M
  iqn_hidden_dim: 32 # cosine final-head width H; monotonic hidden width
  iqn_num_layers: 1 # cosine final-head hidden-layer count L
  monotonic_num_layers: 2
  monotonic_activation: tanh
```

For the monotonic head, `iqn_hidden_dim` is the common width of its hidden
layers and `monotonic_num_layers` is K. An optional
`monotonic_hidden_dims: [32, 16, 8]` overrides both settings when different
layer widths are wanted. Supported monotonic activations are `tanh`,
`sigmoid`, and `softplus`. It ignores `iqn_num_layers`, `cos_emb_dim`, and the
cosine interval settings.

For the cosine head, let `D` be the context width: `dim_model` when current
features are disabled, or `dim_model + current_feature_dim` when they are
enabled. Let `M = cos_emb_dim`, `H = iqn_hidden_dim`, and
`L = iqn_num_layers`. Its architecture is:

```text
tau -> M cosine features -> Linear(M, D) -> ReLU --+
                                                      elementwise multiply
h in R^D ------------------------------------------+
          -> L x [Linear -> ReLU], width H -> Linear(H, 1)
```

The cosine embedding always has exactly one learned linear/ReLU layer;
`iqn_num_layers` counts only the final prediction MLP's hidden layers, not its
output layer. The context is fused as `h * embedding(tau)`, without an input
projection or a residual `1 + embedding(tau)` term. The cosine head itself has
no dropout; `model.dropout` still applies to the RNN or Transformer encoder.
Every final-head hidden layer has width `iqn_hidden_dim`.

For the cosine head, both RNN and Transformer models support two interval
construction modes through `model.interval_mode`:

- `direct` evaluates the raw learned function `g(h, tau)` at every requested
  level. It is deterministic in evaluation mode, but the unconstrained cosine
  head can cross, and no sorting is applied.
- `sampling` draws `model.sampling_num` uniformly distributed levels in one
  call and returns empirical quantiles of the corresponding raw outputs. All
  requested endpoints share that sample, so sorted requested levels produce
  noncrossing endpoints within the call. Test-time inference remains
  stochastic unless the caller controls the random seed.

The default is `interval_mode: sampling` with `sampling_num: 1000`.
`model.sampling_num` controls cosine sampling-based interval construction and
the matching `target_quantiles` validation path. It is separate from
`model.num_taus`, the number of uniformly sampled levels used per example by
the full-range training loss. The partially monotonic head ignores the cosine
depth, embedding, and interval options and always evaluates its nondecreasing
function directly.

Both heads support two training objectives through `training.tau_mode`:

- `sampled_quantiles` (default, including when omitted) retains full
  quantile-function training: each observation uses `model.num_taus` fresh
  uniformly sampled levels and the raw differentiable pinball loss.
- `target_quantiles` evaluates the raw head at every sorted distinct level in
  `model.target_quantiles` for every observation, then averages the same pinball
  loss over observations and levels. It does not sample training levels and
  ignores `model.num_taus` for training. This is a fixed-level diagnostic, not
  supervision of the entire quantile function. The architecture is unchanged.

Training mode is independent of both checkpoint validation and interval
construction. In particular, target-only training with cosine
`interval_mode: sampling` emits a warning but is not rejected or overridden:
sampling still evaluates levels not supervised by that training objective.
For an endpoint-learning diagnostic, use direct intervals and matched
target-level checkpoint validation:

```yaml
model:
  prediction_head: cosine_embedding
  target_quantiles:
    - [0.95, 0.05]
  interval_mode: direct
training:
  tau_mode: target_quantiles
  validation_loss: target_quantiles
```

The target-quantile pairs retain their existing `[upper, lower]` convention;
training uses their sorted distinct endpoints, so the example supervises both
0.05 and 0.95 for every observation. Raw cosine outputs can still cross.

Checkpoint validation is configured separately through
`training.validation_loss`:

- `target_quantiles` averages pinball loss over validation observations and
  the sorted distinct levels in `model.target_quantiles`, using the selected
  interval inference mode. Thus direct cosine validation scores direct
  endpoints, sampling cosine validation scores empirical-rearranged endpoints,
  and partially monotonic validation scores its direct endpoints. Sampling
  validation uses a fixed per-batch seed, `config.seed` (default `0`) plus the
  batch index, and restores the caller's RNG state afterward. Repeating a
  validation epoch therefore scores the same samples without changing later
  training randomness.
- `sampled_quantiles` averages over `model.num_taus` fresh uniformly sampled
  levels and preserves the legacy raw-head validation behavior independently of interval
  construction mode. Configurations that omit `training.validation_loss` also
  fall back to `sampled_quantiles` for backward compatibility.

The checked-in ordinary IQN configurations default to
`tau_mode: sampled_quantiles`; their checkpoint criterion and the tuning grids
explicitly use `validation_loss: target_quantiles`. To compare training modes
or checkpoint criteria in a grid, add either or both axes:

```yaml
grid:
  training.tau_mode: [sampled_quantiles, target_quantiles]
  training.validation_loss: [target_quantiles, sampled_quantiles]
```

The two head choices have different state-dictionary layouts, so a checkpoint
must be reconstructed with the complete matching model configuration. The
paper-style cosine architecture is also not directly compatible with cosine
checkpoints from the earlier implementation, which used a two-linear-layer
PReLU embedding, a learned context input projection, residual multiplicative
conditioning, and a Softplus/dropout prediction head. Retrain those models.
Within the new cosine architecture, `interval_mode`, `sampling_num`, and
`training.tau_mode` do not add state-dictionary entries or change checkpoint
architecture compatibility, but `iqn_num_layers`, `iqn_hidden_dim`,
`cos_emb_dim`, and the context width must match the saved weights. Each
ordinary run writes `resolved_config.yaml` beside its checkpoints, and the
checked-in monotonic experiments use head-specific `saving_dir` values to
avoid overwriting cosine-head results. Their
output paths interpolate `model.prediction_head`, so changing the selector also
changes the ordinary run directory. For separate tuning invocations, likewise
use a distinct `--save-dir`; a single grid run already keeps its head choices
in distinct trials. Configurations that omit the selector retain the legacy
`cosine_embedding` behavior. The Slurm launchers still use
`IQN_PREDICTION_HEAD` to select the head, overriding the YAML selector; interval
and training modes come from the selected YAML. For the cosine diagnostic,
submit with `IQN_PREDICTION_HEAD=cosine_embedding` as well as setting the YAML
options above. IQN-CP currently supports `model.prediction_step: 1`.

## Data split strategy

### Two-stage chronological split

Data are split independently for every time series or station. The outer
base-predictor split and the CP target split preserve time order; training
windows may be shuffled only after their chronological partition has been
chosen.

The pipeline has two stages:

```text
learned base predictor:
raw series = [base-predictor fit prefix] [saved base-predictor held-out suffix]
                                                   |
                                                   +--> [CP train] [CP validation] [CP calibration, when required] [CP test]

pretrained Chronos:
raw series = [initial context] [saved forecast/held-out suffix]
                                      |
                                      +--> [CP train] [CP validation] [CP calibration, when required] [CP test]
```

Every CP runner operates on a saved base-predictor artifact. QR-CP, IQN-CP,
Local-CP, SPCI, and HopCPT read `heldout_x`, `heldout_y`, and
`heldout_predictions`; NexCP, KOWCPI, and ResCP need only the latter two arrays. None
uses the base predictor's fitting prefix. Consequently, CP split ratios are
fractions of the saved held-out suffix, not fractions of the complete raw
series.

For a learned base predictor with raw-series training fraction `b`, CP train
fraction `r_train`, CP validation fraction `r_valid`, and (where applicable)
CP calibration fraction `r_cal`, the approximate raw data allocation is

```text
base fit:       b
CP train:       (1 - b) * r_train
CP validation:  (1 - b) * r_valid
CP calibration: (1 - b) * r_cal                 # Local-CP only
CP test:        (1 - b) * (1 - r_train - r_valid - r_cal)
```

For methods without a calibration partition, set `r_cal=0` in this
calculation.

Window construction and integer rounding can make the exact counts differ
from these proportions.

### Base-predictor stage

The values below are command-line defaults. An artifact generated with
different arguments retains that earlier split, so its recorded generation
command, saved metadata where available, and array lengths are more
authoritative than the current defaults.

| Base predictor | Raw-series allocation | Data saved for CP |
| --- | --- | --- |
| Chronos-2 | No local fitting split is used because the model is pretrained. With the defaults, each rollout uses `window_length=100` historical target/covariate pairs and predicts a block of `prediction_length=10` without receiving covariates from the forecast block. The origin then advances by 10. Only the median (`0.5` quantile) is requested for evaluation. | Positions from `window_length` to the end, with the block forecasts flattened into one held-out sequence. `prediction_length` is a rollout horizon/stride, not a split percentage. |
| Linear regression | A chronological prefix is fitted per series and the remaining suffix is held out. `--train-ratio` defaults to `0.33`; `--past-window` defaults to `100`. The fitted coefficients remain fixed during evaluation. | The complete held-out suffix and rolling one-step-ahead forecasts. Each true held-out target becomes lagged context only after it is observed. |
| LightGBM | The same prefix/suffix strategy as linear regression. `--train-ratio` defaults to `0.33`; `--past-window` defaults to `50`. | The complete held-out suffix and its point forecasts. LightGBM is implemented but is not used by the current checked-in CP configurations. |
| Ridge regression | A chronological prefix is fitted per series and the remaining suffix is held out. `--train-ratio` defaults to `0.33`. `RidgeCV` selects its penalty using the prefix only. | The complete held-out suffix and its point forecasts. |
| LSTM | Each series is first split into a `0.33` fitting prefix and a `0.67` held-out suffix by default. Every fitting prefix is then split chronologically: its first `0.90` supplies training observations and its last `0.10` supplies validation observations. Global normalization statistics use only the pooled inner-training portions. One-step training and validation windows are pooled separately, and the checkpoint with the lowest mean per-sequence validation MSE is selected. | The complete held-out suffix and rolling one-step-ahead forecasts. The selected weights and normalization statistics remain fixed. Each true held-out target becomes lagged context only after it is observed; predictions are not fed back. |

Implementations and launch arguments are in
[`base_predictor/`](base_predictor/) and
[`sbatch_run_base_predictor/`](sbatch_run_base_predictor/). In particular,
Chronos rolling origins are defined in
[`chronos_predictor.py`](base_predictor/chronos_predictor.py#L28-L90), while
the learned-model defaults are defined by the corresponding `run_*.py`
launchers.

### Sapflow (`sapflux-solo3-large`)

The paper-faithful Sapflow input is loaded from
`data/sapflux/0.1.5/prepared/solo_3`. The `large` variant means that
every CSV with an inclusive length of 15,000--20,000 observations is retained;
with the bundled data this selects all 24 paper series. Rows are kept in their
original order and are not resampled, so the native 10-, 15-, and 60-minute
cadences remain distinct.

Each retained series has scalar `y` and ten environmental columns in `x`:
`ta`, `rh`, `sw_in`, `ppfd_in`, `ws`, `precip`, `swc_shallow`, `swc_deep`,
`ext_rad`, and `vpd`. The loader validates the timestamp/schema and emits
`x.shape == (T, 10)` and `y.shape == (T,)` as `float32` arrays.

Generate the recommended global LSTM base-predictor artifact from the
repository root with:

```text
python -m sbatch_run_base_predictor.run_lstm sapflux-solo3-large --max-epoch 50 --device 0
```

The artifact is written to
`data/sapflux-solo3-large/lstm/lstm_sapflux-solo3-large_data.pkl`. Linear
regression, ridge, LightGBM, and Chronos launchers accept the same dataset ID;
their normal `--data-dir` and `--save-dir` overrides remain available. LSTM is
used by the checked-in Sapflow configurations because it supports unequal
series lengths and matches the global-LSTM setup considered in the source
paper.

Ready-to-run Sapflow configurations are provided for QR-CP, IQN-CP, Local-CP,
HopCPT, SPCI, NexCP, and KOWCPI. For example:

```text
python -m sbatch_run_qr_cp.run_rnn_qr_cp configs/qr_cp_configs/qr_rnn_lstm_sapflux_config.yaml
python -m sbatch_run_iqn_cp.run_rnn_iqn_cp configs/iqn_cp_configs/iqn_rnn_lstm_sapflux_config.yaml
python -m sbatch_run_local_cp.run_rnn_local_cp configs/lcp_configs/lcp_rnn_lstm_sapflux_config.yaml
python -m sbatch_run_hopcpt.run_hopcpt configs/hopcpt_configs/hopcpt_lstm_sapflux_config.yaml
python -m sbatch_run_spci.run_spci configs/spci_configs/spci_lstm_sapflux_config.yaml
python -m sbatch_run_nexcp.run_nexcp configs/nexcp_configs/nexcp_lstm_sapflux_config.yaml
python -m sbatch_run_kowcpi.run_kowcpi configs/kowcpi_configs/kowcpi_lstm_sapflux_config.yaml
```

RNN and Transformer variants are supplied for QR-CP, IQN-CP, and Local-CP.
Configurations that support normalization enable it using the appropriate
training prefix, and keep `use_current_feature: False` where that option
exists. Ordinary per-series HopCPT supports the unequal lengths; its
sequence-batch runner currently requires equal-length memories and should not
be used for this dataset.

### Common CP split construction

QR-CP, IQN-CP, and Local-CP call
[`ConformalPredictionData.prepare_quantile_regression_datasets`](dscp/data.py).
The ordinary QR-CP and IQN-CP runners use its legacy three-way split;
the nested QR-CP, IQN-CP, and Local-CP tuning variants are described below.
SPCI uses a training prefix defined by `data.train_ratio` and a final test
suffix, with its tuning validation tail described in the SPCI section below.
For a saved held-out sequence of length `L`, the legacy split computes

```python
n_train = floor(L * train_ratio)
n_valid = ceil(L * valid_ratio)
n_test = L - n_train - n_valid
```

The ordinary Local-CP runner supplies all four ratios and uses cumulative
chronological boundaries:

```python
train_end = floor(L * train_ratio)
valid_end = floor(L * (train_ratio + valid_ratio))
calibration_end = floor(
    L * (train_ratio + valid_ratio + calibration_ratio)
)

n_train = train_end
n_valid = valid_end - train_end
n_calibration = calibration_end - valid_end
n_test = L - calibration_end
```

The four Local-CP ratios must be finite, strictly positive, and sum to one.
The cumulative-boundary calculation makes integer rounding deterministic and
uses every held-out observation.

The sequence is normalized, when enabled, using statistics from only the
first `n_train` observations. Signed prediction residuals are then computed
as

```text
residual_t = heldout_y_t - heldout_prediction_t
```

For history length `W` and prediction horizon `P`, the constructor creates
`L - W - P + 1` aligned examples before applying fixed target sizes to every
partition after training. For the three-way split, the resulting counts are

```text
CP train examples:       n_train - W - P + 1
CP validation examples:  n_valid
CP test examples:        n_test
```

Local-CP currently requires `P=1`; its four-way counts are

```text
CP train examples:        n_train - W
CP validation examples:   n_valid
CP calibration examples:  n_calibration
CP test examples:         n_test
```

Thus, with the currently configured `P=1`, the first `W` held-out
observations supply history rather than training targets. Context windows at
a split boundary may contain observations from the immediately preceding
split; they never contain the current or a future response/residual target.
When `use_current_feature` is enabled, the separately supplied covariate
`x_t` is assumed to be known at prediction time.
This is a downstream CP option and does not change the information supplied
to the Chronos base predictor.

### QR-CP, IQN-CP, and Local-CP data usage

The following table describes how the checked-in ordinary (non-tuning)
air-data runners use the base-predictor held-out suffix. Percentages are
approximate because split boundaries are integer-valued.

| Method | Training | Validation | Calibration | Test |
| --- | --- | --- | --- | --- |
| QR-CP | First 50%. Fits the shared encoder and fixed quantile heads with quantile loss. | Next 16%. Selects the checkpoint and controls early stopping. It does not update model parameters. | None. QR-CP directly estimates conditional residual quantiles and has no separate conformal calibration step. | Final approximately 34%. The frozen checkpoint produces residual quantiles used for coverage, width, and Winkler-score reporting. |
| IQN-CP | First 60%. Fits the shared encoder and tau-conditioned IQN head with sampled quantile loss by default, or fixed target-level loss when `training.tau_mode: target_quantiles`. | Next 20%. Selects the checkpoint and controls early stopping. It does not update model parameters. | None. IQN-CP directly estimates conditional residual quantiles and has no separate conformal calibration step. | Final approximately 20%. The frozen checkpoint produces residual quantiles used for reporting. |
| Local-CP | First 60%. Fits the encoder and auxiliary residual quantile head with quantile loss at `model.training_quantiles`; the hidden state supplies the similarity representation. | Next approximately 10%. Uses quantile loss at the same `model.training_quantiles` to select the checkpoint and control early stopping; it is not used for calibration. | Next approximately 10%. This is a dedicated calibration partition that is not used for fitting or checkpoint selection. The latest `model.calibration_size=500` eligible examples initialize the calibration pool; set a different cap to control its memory size. Each representation for time `t` is paired with its target residual `r_t`. | Final approximately 20%, processed sequentially with batch size one. The network stays frozen. With `model.rolling_calibration: true` (default), `(representation_t, r_t)` replaces the oldest pool pair after the interval is constructed and `r_t` is observed. With `false`, the initial pool is retained throughout inference. |

Thus, `data.calibration_ratio` controls how much data is reserved and eligible
for Local-CP calibration, while `model.calibration_size` caps how many of the
latest eligible points are retained in the calibration pool. If the
dedicated partition contains fewer points than the cap, all of them are used.
The checked-in RNN and Transformer air configurations use
`0.6 / 0.1 / 0.1 / 0.2` for training, validation, calibration, and test,
respectively, with a pool cap of 500.

### Local-CP calibration pool updates

Both RNN and Transformer runners, including tuning, accept:

```yaml
model:
  calibration_size: 500
  rolling_calibration: true
```

- `true` (the default, also when the setting is omitted): after predicting
  each point and observing its outcome, add its representation/residual pair
  and remove the oldest pair, maintaining the initialized pool size.
- `false`: retain the initially selected calibration pairs for every
  inference point; observed evaluation residuals do not enter the pool.

`model.calibration_size` applies in both modes: initialization selects the
latest eligible calibration pairs, up to the cap. The encoder stays frozen
and similarity weights are recomputed for each query in either mode.
This setting controls membership of the calibration pool. Sequential
historical contexts can still include past observed outcomes in both modes.

The tuning grids include a commented `model.rolling_calibration: [true, false]`
line. Uncomment it to compare the two modes during tuning; otherwise, tuning
uses the ordinary configuration's setting.

### Local-CP encoder training quantiles

RNN and Transformer Local-CP train their encoder using an auxiliary residual
quantile head. Configure its fixed levels independently from the interval
quantiles used for inference and evaluation:

```yaml
model:
  training_quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]
  target_quantiles:
    - [0.05, 0.95]
```

`model.training_quantiles` must be a flat, nonempty list of distinct, finite
numbers strictly between zero and one. Levels are sorted into a canonical
order. A single level is supported; the supplied configurations use multiple
levels so training can capture more of the conditional residual distribution.
Ordinary and tuning runners use the same quantile loss for training and
checkpoint validation.

After checkpoint selection, the encoder stays fixed and the auxiliary
quantile head is discarded from the inference procedure. Local-CP samples
calibration residuals using representation-similarity weights, then computes
the final quantiles at `model.target_quantiles` from those shared samples.
Changing inference levels therefore does not change the training objective.
The tuning grids include a commented `model.training_quantiles` example;
each nested list represents one candidate set of training levels.

Previous Local-CP point-predictor checkpoints require retraining because the
quantile head and stored training-level buffer change the checkpoint format.

### Local-CP calibration weights

Choose `model.similarity_fn` to compute weights for a query representation
`h` and calibration representations `h_i`. All options apply softmax across
the calibration pool, giving nonnegative weights that sum to one.

| `similarity_fn` | Score passed to softmax | Effect of increasing `temperature` |
| --- | --- | --- |
| `dot_product` | `temperature * dot(h, h_i)` | Concentrates weight on larger raw dot products. |
| `cos_similarity` | `temperature * cosine_similarity(h, h_i)` | Concentrates weight on more aligned representations. |
| `euclidean` | `-sum((h - h_i) ** 2) / temperature` | Spreads weight more evenly across calibration points. |

The Euclidean option uses **squared Euclidean distance** on the unnormalized
representations and requires a finite, strictly positive scalar temperature.
Its weights are proportional to `exp(-distance_squared / temperature)`;
smaller temperatures concentrate weight on nearby representations. The dot
product and cosine options retain their existing inverse-temperature
convention, multiplying similarity by `temperature`.

For either the RNN or Transformer runner, configure Euclidean weights with:

```yaml
model:
  similarity_fn: "euclidean"
  temperature: 1.0
```

Both Local-CP tuning grids include all three options. These calibration
settings are separate from `model.training_quantiles` (encoder training)
and `model.target_quantiles` (inference and evaluation).

### Additional CP baseline allocations

| Method | Split of the base-predictor held-out suffix | How each portion is used |
| --- | --- | --- |
| SPCI | 66% train / about 34% test | `data.train_ratio` defines the full training prefix. Tuning reserves its last `model_selection_valid_ratio` fraction for evaluation; final runs fit normalization and one quantile forest on the full prefix. Its parameters stay fixed during test, although each test feature contains the available past residual window. |
| HopCPT | 33% train / 33% validation / about 34% test | Train fits the Hopfield network. Validation is evaluated sequentially and selects the checkpoint by coverage and interval width. The selected network stays fixed in test, while its available context/residual memory advances through validation and prior test observations. |
| NexCP | 66% initial calibration / about 34% test | There is no learned train/validation stage. The initial prefix provides residual history; with the current `max_past=200`, each interval uses at most the latest 200 available residuals, including prior test residuals as testing advances. |
| KOWCPI | 50% nominal train / 16% nominal validation / about 34% test | Train and validation are combined into a 66% initial calibration prefix. With the current `update_with_test=true`, later intervals also use prior test residuals. If normalization is enabled, only the nominal training prefix determines its statistics. |
| ResCP | 66% calibration / about 34% test | Fixed Air/Solar runs initialize on full calibration. Sapflux tuning reserves the last 20% of calibration for validation; final runs initialize afresh and fit normalization on full calibration. Memory updates after each evaluated observation. Both outer ratios and the tuning validation fraction are configurable. |

The active ratios and method-specific settings are in:

- [`configs/qr_cp_configs/`](configs/qr_cp_configs/)
- [`configs/iqn_cp_configs/`](configs/iqn_cp_configs/)
- [`configs/lcp_configs/`](configs/lcp_configs/)
- [`configs/spci_configs/`](configs/spci_configs/)
- [`configs/hopcpt_configs/`](configs/hopcpt_configs/)
- [`configs/nexcp_configs/`](configs/nexcp_configs/)
- [`configs/kowcpi_configs/`](configs/kowcpi_configs/)
- [`configs/rescp_configs/`](configs/rescp_configs/)

Non-air configurations may intentionally use different ratios. For example,
the HopCPT toy configuration uses 25% train / 25% validation / 50% test, and
the KOWCPI simulation configuration uses 80% initial calibration / 20% test.

### Test-time information protocol

These experiments use a one-step online, or prequential, protocol. When a
method uses past residuals, the interval for test time `t` may use test
residuals observed before `t`. This is valid only when outcomes arrive between
successive forecasts. It is not the same as producing an entire multi-step
test block simultaneously.

Chronos `prediction_length` and CP `model.prediction_step` are independent.
The Chronos artifact used by the current air configurations was generated with
ten-step rollout blocks, while the current CP runners operate one timestamp at
a time (`model.prediction_step=1` where configurable). Since the saved Chronos
data flatten and discard lead indices, those CP datasets mix base-forecast
leads 1 through 10. Use Chronos
`--prediction-length 1` when homogeneous rolling one-step base-forecast errors
are required.

For artifacts generated by the current Chronos implementation, every lead in
a block is forecast using only target and covariate rows from the preceding
context window; covariates at the predicted timestamps are not supplied. At
the next block origin, observations from the preceding block have become
historical and enter the new context window. Existing Chronos artifacts must
be regenerated to adopt this past-only covariate protocol.

### Selecting the Air base predictor for QR-CP tuning

Each Air tuning job uses a three-task Slurm array, with one task per base
predictor. The RNN and Transformer scripts select the corresponding
`configs/qr_cp_configs/qr_<encoder>_<predictor>_air_config.yaml`:

| Array task | Predictor |
| --- | --- |
| `0` | LR (`lr`) |
| `1` | LSTM (`lstm`) |
| `2` | Chronos (`chronos`) |

Each base configuration declares its predictor at the top level. Keep this
value consistent with the filename when running the arrays:

```yaml
base_predictor: chronos # choices: lr, lstm, chronos
```

The base configuration's `data.data_path` and `saving_dir` refer to
`${base_predictor}`, so changing this field updates their predictor component
automatically. For example, `base_predictor: lstm` reads
`data/air-10_prediction/lstm/lstm_air-10_data.pkl`; generate the selected
predictor's artifact before submitting. Submit both arrays to run six separate
tuning tasks, using the same tuning grid for all three predictors of each
encoder:

```bash
sbatch sbatch/sbatch_run_tuning/run_qr_cp_rnn_air_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_qr_cp_transformer_air_tuning.sbatch
```

To submit only one predictor, override the array range. For example, this runs
only RNN/LSTM tuning:

```bash
sbatch --array=1 sbatch/sbatch_run_tuning/run_qr_cp_rnn_air_tuning.sbatch
```

The job's `--base-config` selects the experiment configuration. Changing
`base_predictor` keeps the model and training defaults from that file; the
selected tuning grid overrides only its listed candidate settings. Keep the
candidate hyperparameter values under `grid` and controls such as
`tuning.num_sequences` in the tuning YAML.

The tuning jobs automatically separate results under
`results/tuning/qr_<encoder>_<predictor>_air/`, where `<encoder>` is `rnn` or
`transformer`. The predictor comes from the selected base configuration. Slurm
logs use `Report-%x-%A_%a.out`, separating the job name, array job ID, and task
ID.

### QR-CP tuning on Solar and Sapflux

Solar and Sapflux use the same predictor arrays as Air: task `0` selects LR,
task `1` selects LSTM, and task `2` selects Chronos. Their RNN and Transformer
tuning grids match the corresponding Air grids (48 and 64 combinations).

```bash
sbatch sbatch/sbatch_run_tuning/run_qr_cp_rnn_solar_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_qr_cp_transformer_solar_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_qr_cp_rnn_sapflux_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_qr_cp_transformer_sapflux_tuning.sbatch
```

Each task loads `configs/qr_cp_configs/qr_<encoder>_<predictor>_<dataset>_config.yaml`
and `configs/qr_cp_configs/qr_<encoder>_<dataset>_tuning_config.yaml`.
Solar predictions come from `data/solar_prediction/<predictor>/<predictor>_nsdb-60m_data.pkl`;
Sapflux predictions come from
`data/sapflux-solo3-large/<predictor>/<predictor>_sapflux-solo3-large_data.pkl`.
Generate the selected base predictor artifacts before submitting.
Results are saved under `results/tuning/qr_<encoder>_<predictor>_<dataset>/`.

All four launchers follow the current Air templates: two GPUs per array task,
two concurrent trial workers, and a four-hour time limit. RNN tasks request
two CPUs and 8 GB RAM; Transformer tasks request four CPUs and 16 GB RAM.
The scripts pass `--num-gpus 2`, overriding the tuning YAML default of one.
To run a single predictor on one GPU, for example:

```bash
sbatch --array=1 --gres=gpu:1 --cpus-per-task=2 --mem=8G \
  sbatch/sbatch_run_tuning/run_qr_cp_transformer_solar_tuning.sbatch --num-gpus 1
```

### Selecting the Air base predictor for IQN-CP tuning

IQN-CP uses the same three-task predictor arrays as QR-CP: task `0` selects
LR, task `1` selects LSTM, and task `2` selects Chronos. Submit both encoders
to run six separate tuning tasks:

```bash
sbatch sbatch/sbatch_run_tuning/run_iqn_cp_rnn_air_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_iqn_cp_transformer_air_tuning.sbatch
```

Each task loads
`configs/iqn_cp_configs/iqn_<encoder>_<predictor>_air_config.yaml` and uses the
encoder's existing tuning grid. Each base configuration declares
`base_predictor` and interpolates it into `data.data_path` and the ordinary
run's `saving_dir`; the ordinary output also retains its prediction-head
component. Generate the selected predictor's saved forecast artifact before
submitting. The generic `iqn_<encoder>_air_config.yaml` configurations remain
available and default to Chronos for compatibility.

Tuning results are separated under
`results/tuning/iqn_<encoder>_<predictor>_air/`, with the predictor resolved
from the actual base configuration. Trial configurations preserve their
prediction-head settings, and both encoders keep their existing tuning
grids. Slurm logs use `Report-%x-%A_%a.out` to separate array tasks.

To tune only LSTM with the RNN encoder, select task `1`. To run all predictors
with at most one task active at a time, limit array concurrency:

```bash
sbatch --array=1 sbatch/sbatch_run_tuning/run_iqn_cp_rnn_air_tuning.sbatch
sbatch --array=0-2%1 sbatch/sbatch_run_tuning/run_iqn_cp_transformer_air_tuning.sbatch
```

Each IQN-CP task requests two GPUs, two CPUs, 4 GB of host memory, and six
hours, and passes `--num-gpus 2` to run two hyperparameter trials concurrently.
Arguments after the script name can override the worker count. One full
encoder array can use six GPUs, or both arrays can use twelve GPUs when all
tasks run concurrently. Adjust the Slurm CPU, memory, and time limits for
the search and data size.

### Selecting the Air base predictor for Local-CP tuning

Local-CP uses the same three-task predictor arrays as QR-CP: task `0` selects
LR, task `1` selects LSTM, and task `2` selects Chronos. Submit both encoders
to run six separate tuning tasks:

```bash
sbatch sbatch/sbatch_run_tuning/run_lcp_rnn_air_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_lcp_transformer_air_tuning.sbatch
```

Each task loads
`configs/lcp_configs/lcp_<encoder>_<predictor>_air_config.yaml` and uses the
encoder's existing tuning grid. The six base configurations declare
`base_predictor` and interpolate it into `data.data_path` and `saving_dir`.
Generate the selected predictor's saved forecast artifact before submitting.
Model and training defaults are shared across the three predictors for each
encoder; the grid overrides only its listed candidate settings.

Results are separated under
`results/tuning/lcp_<encoder>_<predictor>_air/`, with the predictor resolved
from the actual base configuration. Explicit configurations with an existing
predictor-specific `data.data_path` also remain supported. Slurm logs use
`Report-%x-%A_%a.out` to separate array tasks.

To tune only LSTM with the RNN encoder, select task `1`. To run all predictors
with at most one task active at a time, limit array concurrency:

```bash
sbatch --array=1 sbatch/sbatch_run_tuning/run_lcp_rnn_air_tuning.sbatch
sbatch --array=0-2%1 sbatch/sbatch_run_tuning/run_lcp_transformer_air_tuning.sbatch
```

Each Local-CP task requests two GPUs, two CPUs, 8 GB of host memory, and eight
hours, and runs two hyperparameter trials concurrently. Predictor array tasks
are separate from these GPU workers: one full encoder array can use six GPUs,
or both arrays can use twelve GPUs when all tasks run concurrently. Adjust
the Slurm CPU, memory, and time limits for the search and data size.

### Optional multi-GPU QR-CP tuning

QR-CP can run independent hyperparameter trials on multiple GPUs. Set the
number of GPU workers in the encoder's tuning YAML
([RNN](configs/qr_cp_configs/qr_rnn_air_tuning_config.yaml) or
[Transformer](configs/qr_cp_configs/qr_transformer_air_tuning_config.yaml)):

```yaml
tuning:
  num_gpus: 1 # default; use 2 or more to parallelize trials
```

With `num_gpus: 1`, trials run sequentially on one GPU, with the existing CPU
fallback when CUDA is unavailable. With a larger value, each GPU runs one
worker process and takes the next available trial. Each trial still trains
and evaluates all selected sequences in their existing order. Trial seeds
are assigned independently of worker scheduling, and the parent process
combines all trial results before ranking configurations.

Request the matching GPUs from Slurm as well: a YAML setting cannot allocate
cluster resources. The RNN script requests one GPU per task and uses the YAML
worker count by default. The Transformer script requests two GPUs per task
and passes `--num-gpus 2`, overriding the YAML default. Run both with two GPUs
per task as follows:

```bash
sbatch --gres=gpu:2 --cpus-per-task=4 --mem=16G sbatch/sbatch_run_tuning/run_qr_cp_rnn_air_tuning.sbatch --num-gpus 2
sbatch sbatch/sbatch_run_tuning/run_qr_cp_transformer_air_tuning.sbatch
```

For RNN, you can alternatively set `tuning.num_gpus: 2` and omit the final
`--num-gpus 2` argument while retaining the Slurm resource options. Arguments
after either script name override its worker count. To run Transformer trials
sequentially on one GPU:

```bash
sbatch --gres=gpu:1 --cpus-per-task=2 --mem=8G sbatch/sbatch_run_tuning/run_qr_cp_transformer_air_tuning.sbatch --num-gpus 1
```

The three predictor array tasks remain separate. Two GPUs per task require
up to six GPUs for one encoder's array, or twelve for both arrays when all
tasks run concurrently. Workers have separate data and caches, so host-memory
requirements grow with the worker count; adjust the example's `--mem=16G`
allocation for your data. Multi-GPU mode fails clearly if fewer than the
requested number of CUDA GPUs are visible.

### Optional multi-GPU IQN-CP and Local-CP tuning

IQN-CP and Local-CP use the same per-trial GPU workers and controls as QR-CP.
Their tuning YAMLs default to `tuning.num_gpus: 1`:

| Method | RNN tuning config | Transformer tuning config |
| --- | --- | --- |
| IQN-CP | [RNN](configs/iqn_cp_configs/iqn_rnn_air_tuning_config.yaml) | [Transformer](configs/iqn_cp_configs/iqn_transformer_air_tuning_config.yaml) |
| Local-CP | [RNN](configs/lcp_configs/lcp_rnn_air_tuning_config.yaml) | [Transformer](configs/lcp_configs/lcp_transformer_air_tuning_config.yaml) |

Set `tuning.num_gpus: 2` to run two independent trials concurrently, or override
the YAML with `--num-gpus 2`. The four Slurm launchers currently request two
GPUs and pass `--num-gpus 2`, overriding the YAML default. Arguments after the
script name can override that worker count. For example:

```bash
sbatch --gres=gpu:2 --cpus-per-task=4 --mem=16G sbatch/sbatch_run_tuning/run_iqn_cp_rnn_air_tuning.sbatch --num-gpus 2
sbatch --gres=gpu:2 --cpus-per-task=4 --mem=16G sbatch/sbatch_run_tuning/run_iqn_cp_transformer_air_tuning.sbatch --num-gpus 2
sbatch --cpus-per-task=4 --mem=16G sbatch/sbatch_run_tuning/run_lcp_rnn_air_tuning.sbatch
sbatch --cpus-per-task=4 --mem=16G sbatch/sbatch_run_tuning/run_lcp_transformer_air_tuning.sbatch
```

Both IQN-CP and Local-CP submissions launch one task per predictor. To run
Local-CP trials sequentially on one GPU per predictor task, override both
the Slurm allocation and worker count:

```bash
sbatch --gres=gpu:1 sbatch/sbatch_run_tuning/run_lcp_rnn_air_tuning.sbatch --num-gpus 1
```

The same resource and worker overrides apply to the IQN-CP launchers.
Multi-GPU execution uses the first requested number of
visible CUDA devices and fails before loading data if too few are available.
Single-worker execution retains the existing CPU fallback.

Each worker evaluates every selected sequence for its assigned trial, using
the original trial seed and sequence order. The parent combines all results
and selects the best configurations across the full grid. IQN-CP keeps its
head and shared-dimension settings; Local-CP keeps its separate calibration
partition and sequential rolling updates within each sequence. The reserved
final test partition remains excluded from tuning. Host-memory requirements
increase with the number of workers because each retains its own data cache.

### Selecting sequences for tuning

QR-CP, IQN-CP, and Local-CP accept the same setting in their tuning YAML to
include every available sequence:

```yaml
tuning:
  num_sequences: all
```

`all` selects every sequence in sorted key order. A positive integer selects
that many consecutive sorted keys starting at `--sequence-index` (default
`0`). For example, `num_sequences: 2` selects two sequences. An explicit
`--sequence-key KEY` takes precedence and selects only that sequence, including
when `num_sequences: all`. With `all` and no explicit key, `--sequence-index`
is ignored.

### SPCI hyperparameter tuning

SPCI uses the same grid, sequence-selection, coverage-filtering, and Winkler-score
ranking conventions as QR-CP. Submit the CPU arrays for LR/LSTM/Chronos with:

```bash
sbatch sbatch/sbatch_run_tuning/run_spci_air_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_spci_solar_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_spci_sapflux_tuning.sbatch
```

In the SPCI tuning YAML, `tuning.model_selection_valid_ratio` controls the
**fraction of the training prefix reserved for hyperparameter evaluation**.
`data.train_ratio` defines the full training prefix, and the remaining suffix
is the final test set. For example, with `data.train_ratio: 0.66` and
`tuning.model_selection_valid_ratio: 0.2`, the first 52.8% of the saved
predictor's held-out sequence fits the forest, the next 13.2% evaluates
candidates, and the last 34% remains reserved for test. Normalization
statistics come only from the fitting prefix. There is no separate outer
validation region. The shipped grids use `model_selection_valid_ratio: 0.15`,
giving approximately 56.1% fitting and 9.9% tuning evaluation. SPCI uses this
subset to rank forest hyperparameters; it has no epoch or checkpoint-selection
stage.

For an older SPCI config, replace `data.train_ratio` with the sum of its old
`train_ratio` and `valid_ratio`, then remove `data.valid_ratio`. The checked-in
dataset configs now use `0.66` (previously `0.5 + 0.16`), and the toy config
uses `0.8` (previously `0.6 + 0.2`). This preserves approximately the same
final test boundary, with a possible one-observation shift from integer
rounding. The runner rejects configs that still contain `data.valid_ratio`.
Final evaluation refits on the full training prefix, including the tuning
validation tail, and evaluates the remaining test suffix. Normalization now
uses the full training prefix instead of the old nominal training portion,
so migration can change the results.

The grids search residual-window length, tree count, and tree depth. Each trial
must pass the coverage threshold on every selected sequence and confidence pair;
eligible trials are ranked by mean Winkler score. Results include per-trial
resolved YAMLs and `tuning_results.pkl`, with the same `all_trials` and `top_trials`
structure as QR-CP. Run a selected resolved YAML through the ordinary SPCI runner
for final refitting and test evaluation. See the
[SPCI job documentation](sbatch/sbatch_run_spci/README.md#hyperparameter-tuning)
for grids, overrides, the exact split, and final-run commands.

### Hyperparameter-tuning protocol

QR-CP, IQN-CP, and Local-CP grid search use nested chronological splits so
the final test suffix is not used for hyperparameter selection. The setting
`tuning.model_selection_valid_ratio: 0.2` assigns the later 20% of the
nominal training prefix to checkpoint selection. The earlier 80% fits the
model. QR-CP and IQN-CP then use the nominal validation partition to evaluate
and rank trials.

For the checked-in air configurations, the allocations below are percentages
of the saved base-predictor held-out suffix. They are approximate only when
integer rounding is required.

| Tuning role | QR-CP (`50 / 16 / 34` outer split) | IQN-CP (`60 / 20 / 20` outer split) | Use |
| --- | --- | --- | --- |
| Inner model fit | First 40% (80% of the nominal 50% training prefix) | First 48% (80% of the nominal 60% training prefix) | Optimizes model parameters. When normalization is enabled, only this prefix determines its statistics. |
| Inner checkpoint validation | Next 10% (later 20% of nominal training) | Next 12% (later 20% of nominal training) | Selects the checkpoint and controls early stopping; it does not update parameters. |
| Tuning evaluation | Next 16% (the nominal validation partition) | Next 20% (the nominal validation partition) | Filters trials by coverage and ranks eligible trials by Winkler score. |
| Final test | Final 34% | Final 20% | Remains untouched: no test dataset is requested or consumed by trial training, checkpoint selection, evaluation, filtering, or ranking. |

Local-CP must additionally keep calibration separate from fitting,
checkpoint selection, and tuning evaluation. Its ordinary outer split remains
`0.60 / 0.10 / 0.10 / 0.20`, but the tuner assigns those regions as follows:

| Tuning role | Local-CP allocation | Use |
| --- | --- | --- |
| Inner model fit | First 48% (80% of the outer 60% training prefix) | Optimizes the encoder and auxiliary residual quantile head with quantile loss at `model.training_quantiles`. When normalization is enabled, only this prefix determines its statistics. |
| Inner checkpoint validation | Next 12% (later 20% of the outer training prefix) | Uses quantile loss at the same `model.training_quantiles` to select the checkpoint and control early stopping; it does not update parameters. |
| Tuning calibration | Next 10% (the ordinary runner's validation region) | Initializes Local-CP after the selected model is frozen. `model.calibration_size` still caps the latest eligible calibration points retained in the pool. |
| Tuning evaluation | Next 10% (the ordinary runner's calibration region) | Evaluates coverage and Winkler score and filters/ranks trials. With `model.rolling_calibration: true` (default), updates the pool only after scoring each point and observing its outcome; with `false`, retains the initial tuning calibration pool. |
| Final test | Final 20% | Remains unavailable to the tuner: no test dataset is constructed, requested, or consumed. |

For a length-100 held-out sequence, before accounting for the history window,
the exact Local-CP tuning boundaries are therefore
`[0,48) / [48,60) / [60,70) / [70,80) / [80,100)`. This differs deliberately
from the ordinary Local-CP runner, which uses
`[0,60) / [60,70) / [70,80) / [80,100)` for fit, checkpoint validation,
calibration, and final test.

With a history window of length `W`, the first `W` observations in the inner
fit prefix provide context and are not prediction targets. Context at a later
boundary may include already observed values from the preceding partition,
but never the current or a future target. The nested tuning split currently
supports `model.prediction_step=1`.

After choosing hyperparameters, copy them into the corresponding ordinary
QR-CP, IQN-CP, or Local-CP configuration and run the non-tuning runner. That
runner may then refit using its normal training/validation/calibration
protocol and evaluate the previously untouched final test suffix once.
Grid-search output itself is a validation result, not the final test result.

This nested protocol currently applies to the QR-CP, IQN-CP, and Local-CP
tuning runners. ResCP reserves a validation tail within its nominal calibration
prefix and likewise excludes the final test suffix during tuning. Legacy tuners under
[`sbatch/sbatch_run_tuning/`](sbatch/sbatch_run_tuning/) may still evaluate or
rank candidates on their test partition and should not be assumed to provide
an untouched final evaluation. All methods in a final comparison should use
the same base-predictor artifact and compatible CP split boundaries.
