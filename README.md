# DSCP

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
| `cosine_embedding` (legacy default) | Preserves the original cosine quantile embedding and sampling-based rearrangement behavior, including compatibility with existing configurations and checkpoints. The raw embedded head itself is not constrained to be monotonic. |

The checked-in IQN-CP experiment configurations explicitly select the
partially monotonic design. To switch back to the legacy embedding design,
change only the selector:

```yaml
model:
  prediction_head: partially_monotonic # or cosine_embedding
  iqn_hidden_dim: 32
  monotonic_num_layers: 2
  monotonic_activation: tanh
```

For the monotonic head, `iqn_hidden_dim` is the common width of its hidden
layers and `monotonic_num_layers` is K. An optional
`monotonic_hidden_dims: [32, 16, 8]` overrides both settings when different
layer widths are wanted. Supported monotonic activations are `tanh`,
`sigmoid`, and `softplus`. The `cos_emb_dim` setting is used only by the
legacy cosine head.

The two head choices have different state-dictionary layouts, so a checkpoint
must be reconstructed with the complete matching model configuration. Each
ordinary run now writes `resolved_config.yaml` beside its checkpoints, and the
checked-in monotonic experiments use head-specific `saving_dir` values to
avoid overwriting legacy cosine-head results. Their output paths interpolate
`model.prediction_head`, so changing the selector also changes the ordinary
run directory. For separate tuning invocations, likewise use a distinct
`--save-dir`; a single grid run already keeps its head choices in distinct
trials. Configurations that omit the selector retain the legacy
`cosine_embedding` behavior. IQN-CP currently supports
`model.prediction_step: 1`.

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
`heldout_predictions`; NexCP and KOWCPI need only the latter two arrays. None
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

QR-CP, IQN-CP, Local-CP, and SPCI call
[`ConformalPredictionData.prepare_quantile_regression_datasets`](dscp/data.py).
The ordinary QR-CP, IQN-CP, and SPCI runners use its legacy three-way split;
the nested QR-CP, IQN-CP, and Local-CP tuning variants are described below.
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
| IQN-CP | First 60%. Fits the shared encoder and tau-conditioned IQN head with sampled quantile loss. | Next 20%. Selects the checkpoint and controls early stopping. It does not update model parameters. | None. IQN-CP directly estimates conditional residual quantiles and has no separate conformal calibration step. | Final approximately 20%. The frozen checkpoint produces residual quantiles used for reporting. |
| Local-CP | First 60%. Fits a residual point predictor whose hidden state supplies the similarity representation. | Next approximately 10%. Selects the checkpoint and controls early stopping; it is not used for calibration. | Next approximately 10%. This is a dedicated calibration partition that is not used for fitting or checkpoint selection. The latest `model.calibration_size=500` eligible examples initialize the rolling pool; set a different cap to control its memory size. Each representation for time `t` is paired with its target residual `r_t`. | Final approximately 20%, processed sequentially with batch size one. The network stays frozen. The interval for time `t` is constructed before observing `r_t`; afterward, `(representation_t, r_t)` enters the rolling pool and the oldest point is discarded when the cap is reached. |

Thus, `data.calibration_ratio` controls how much data is reserved and eligible
for Local-CP calibration, while `model.calibration_size` caps how many of the
latest eligible points are retained in the initial and rolling pools. If the
dedicated partition contains fewer points than the cap, all of them are used.
The checked-in RNN and Transformer air configurations use
`0.6 / 0.1 / 0.1 / 0.2` for training, validation, calibration, and test,
respectively, with a pool cap of 500.

### Additional CP baseline allocations

| Method | Split of the base-predictor held-out suffix | How each portion is used |
| --- | --- | --- |
| SPCI | 50% train / 16% validation / about 34% test | The nominal train and validation examples are combined to fit one quantile random forest. Its parameters stay fixed during test, although each test feature contains the available past residual window. |
| HopCPT | 33% train / 33% validation / about 34% test | Train fits the Hopfield network. Validation is evaluated sequentially and selects the checkpoint by coverage and interval width. The selected network stays fixed in test, while its available context/residual memory advances through validation and prior test observations. |
| NexCP | 66% initial calibration / about 34% test | There is no learned train/validation stage. The initial prefix provides residual history; with the current `max_past=200`, each interval uses at most the latest 200 available residuals, including prior test residuals as testing advances. |
| KOWCPI | 50% nominal train / 16% nominal validation / about 34% test | Train and validation are combined into a 66% initial calibration prefix. With the current `update_with_test=true`, later intervals also use prior test residuals. If normalization is enabled, only the nominal training prefix determines its statistics. |

The active ratios and method-specific settings are in:

- [`configs/qr_cp_configs/`](configs/qr_cp_configs/)
- [`configs/iqn_cp_configs/`](configs/iqn_cp_configs/)
- [`configs/lcp_configs/`](configs/lcp_configs/)
- [`configs/spci_configs/`](configs/spci_configs/)
- [`configs/hopcpt_configs/`](configs/hopcpt_configs/)
- [`configs/nexcp_configs/`](configs/nexcp_configs/)
- [`configs/kowcpi_configs/`](configs/kowcpi_configs/)

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
| Inner model fit | First 48% (80% of the outer 60% training prefix) | Optimizes the residual predictor. When normalization is enabled, only this prefix determines its statistics. |
| Inner checkpoint validation | Next 12% (later 20% of the outer training prefix) | Selects the checkpoint and controls early stopping; it does not update parameters. |
| Tuning calibration | Next 10% (the ordinary runner's validation region) | Initializes Local-CP after the selected model is frozen. `model.calibration_size` still caps the latest eligible calibration points retained in the rolling pool. |
| Tuning evaluation | Next 10% (the ordinary runner's calibration region) | Evaluates coverage and Winkler score, updates the rolling pool only after scoring each point, and filters/ranks trials. |
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
tuning runners. Other tuners under
[`sbatch/sbatch_run_tuning/`](sbatch/sbatch_run_tuning/) may still evaluate or
rank candidates on their test partition and should not be assumed to provide
an untouched final evaluation. All methods in a final comparison should use
the same base-predictor artifact and compatible CP split boundaries.
