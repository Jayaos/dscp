# DSCP

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
                                                   +--> [CP train] [CP validation] [CP test]

pretrained Chronos:
raw series = [initial context] [saved forecast/held-out suffix]
                                      |
                                      +--> [CP train] [CP validation] [CP test]
```

Every CP runner operates on a saved base-predictor artifact. QR-CP, IQN-CP,
Local-CP, SPCI, and HopCPT read `heldout_x`, `heldout_y`, and
`heldout_predictions`; NexCP and KOWCPI need only the latter two arrays. None
uses the base predictor's fitting prefix. Consequently, CP split ratios are
fractions of the saved held-out suffix, not fractions of the complete raw
series.

For a learned base predictor with raw-series training fraction `b`, CP train
fraction `r_train`, and CP validation fraction `r_valid`, the approximate raw
data allocation is

```text
base fit:       b
CP train:       (1 - b) * r_train
CP validation:  (1 - b) * r_valid
CP test:        (1 - b) * (1 - r_train - r_valid)
```

Window construction and integer rounding can make the exact counts differ
from these proportions.

### Base-predictor stage

The values below are command-line defaults. An artifact generated with
different arguments retains that earlier split, so its recorded generation
command, saved metadata where available, and array lengths are more
authoritative than the current defaults.

| Base predictor | Raw-series allocation | Data saved for CP |
| --- | --- | --- |
| Chronos-2 | No local fitting split is used because the model is pretrained. With the defaults, each rollout uses `window_length=100` observed values as context and predicts a block of `prediction_length=10`. The origin then advances by 10. | Positions from `window_length` to the end, with the block forecasts flattened into one held-out sequence. `prediction_length` is a rollout horizon/stride, not a split percentage. |
| Linear regression | A chronological prefix is fitted per series and the remaining suffix is held out. `--train-ratio` defaults to `0.33`; `--past-window` defaults to `100`. | The complete held-out suffix and its point forecasts. |
| LightGBM | The same prefix/suffix strategy as linear regression. `--train-ratio` defaults to `0.33`; `--past-window` defaults to `50`. | The complete held-out suffix and its point forecasts. LightGBM is implemented but is not used by the current checked-in CP configurations. |
| Ridge regression | A chronological prefix is fitted per series and the remaining suffix is held out. `--train-ratio` defaults to `0.33`. `RidgeCV` selects its penalty using the prefix only. | The complete held-out suffix and its point forecasts. |
| LSTM | Each series is first split into a `0.33` fitting prefix and a `0.67` held-out suffix by default. Windows from all fitting prefixes are pooled; the first `0.90` of that pooled array trains the global LSTM and the last `0.10` validates/early-stops it. The two subsets are shuffled after this split. | Prediction begins only after the first `window_length - 1` observations of each held-out suffix have supplied an input window, so those context observations are not CP targets. |

Implementations and launch arguments are in
[`base_predictor/`](base_predictor/) and
[`sbatch_run_base_predictor/`](sbatch_run_base_predictor/). In particular,
Chronos rolling origins are defined in
[`chronos_predictor.py`](base_predictor/chronos_predictor.py#L28-L90), while
the learned-model defaults are defined by the corresponding `run_*.py`
launchers.

### Common CP split construction

QR-CP, IQN-CP, Local-CP, and SPCI all call
[`ConformalPredictionData.prepare_quantile_regression_datasets`](dscp/data.py#L19-L155).
For a saved held-out sequence of length `L`, it computes

```python
n_train = floor(L * train_ratio)
n_valid = ceil(L * valid_ratio)
n_test = L - n_train - n_valid
```

The sequence is normalized, when enabled, using statistics from only the
first `n_train` observations. Signed prediction residuals are then computed
as

```text
residual_t = heldout_y_t - heldout_prediction_t
```

For history length `W` and prediction horizon `P`, the constructor creates
`L - W - P + 1` aligned examples before applying the fixed validation and
test target sizes. The resulting counts are

```text
CP train examples:       n_train - W - P + 1
CP validation examples:  n_valid
CP test examples:        n_test
```

Thus, with the currently configured `P=1`, the first `W` held-out
observations supply history rather than training targets. Context windows at
a split boundary may contain observations from the immediately preceding
split; they never contain the current or a future response/residual target.
When `use_current_feature` is enabled, the separately supplied covariate
`x_t` is assumed to be known at prediction time.

### QR-CP, IQN-CP, and Local-CP data usage

The following table describes how the checked-in air-data runners currently
use the base-predictor held-out suffix. Test percentages are remainders and
are approximate because training uses `floor` and validation uses `ceil`.

| Method | Training | Validation | Calibration | Test |
| --- | --- | --- | --- | --- |
| QR-CP | First 50%. Fits the shared encoder and fixed quantile heads with quantile loss. | Next 16%. Selects the checkpoint and controls early stopping. It does not update model parameters. | None. QR-CP directly estimates conditional residual quantiles and has no separate conformal calibration step. | Final approximately 34%. The frozen checkpoint produces residual quantiles used for coverage, width, and Winkler-score reporting. |
| IQN-CP | First 60%. Fits the shared encoder and tau-conditioned IQN head with sampled quantile loss. | Next 20%. Selects the checkpoint and controls early stopping. It does not update model parameters. | None. IQN-CP directly estimates conditional residual quantiles and has no separate conformal calibration step. | Final approximately 20%. The frozen checkpoint produces residual quantiles used for reporting. |
| Local-CP (current code) | First 60%. Fits a residual point predictor whose hidden state supplies the similarity representation. | Next 20%. Selects the checkpoint and controls early stopping. | The last `calibration_size=500` examples from combined training and validation initialize the pool. With the current sizes these are the final 500 validation examples, so calibration overlaps model selection and is not independent. | Final approximately 20%, processed sequentially with batch size one. The network stays frozen; prior test observations progressively replace the initial calibration points. |

Local-CP requires a calibration set that was not used for either parameter
fitting or checkpoint/hyperparameter selection. A clean layout that preserves
the current final 20% test boundary and fixed calibration size is:

| Method | Training | Validation | Dedicated calibration | Test |
| --- | --- | --- | --- | --- |
| Local-CP (required clean split) | First 60%. Fit the residual representation model. | The following pre-test observations, stopping 500 observations before test. Use only for checkpoint and hyperparameter selection. | The final 500 pre-test observations. Freeze the model before encoding these points, and pair each representation for time `t` with target residual `r_t`. | Final approximately 20%. Construct each interval before observing `r_t`; afterward, add the newly observed `(representation_t, r_t)` pair to the rolling pool. |

The dedicated Local-CP layout is the documented target protocol; the current
data loader and runner do not yet construct it as a separate partition.

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

### Hyperparameter-tuning split caution

Where applicable, the ordinary method runners use validation for checkpoint
selection and test for reporting as described above. The current grid-search
runners under [`sbatch_run_tuning/`](sbatch_run_tuning/) additionally evaluate every
hyperparameter trial on the nominal test portion, filter trials by test
coverage, and rank them by test Winkler score. Their reported test results are
therefore tuning results, not an untouched final evaluation.

For a clean final comparison, choose hyperparameters without the final test
suffix, freeze the chosen configuration, and evaluate that suffix once. All
methods being compared should also use the same base-predictor artifact and
the same CP split boundaries.

### Current Local-CP alignment caveat

The intended Local-CP calibration pair is the representation used to predict
target residual `r_t` together with `r_t`. The current encoder export and
online update instead attach the last input residual `r_(t-1)` to that
representation. This one-step label shift occurs in both the initial and
rolling calibration pools; see
[`rnn_predictor.py`](dscp/models/rnn_predictor.py#L62-L85),
[`transformer_predictor.py`](dscp/models/transformer_predictor.py#L61-L83),
and [`run_local_cp.py`](dscp/run_local_cp.py#L166-L283). It should be resolved
before interpreting Local-CP results as implementing the intended calibration
strategy.
