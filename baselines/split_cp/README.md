# SplitCP

Standalone split conformal prediction for saved DSCP base forecasts, using
Tibshirani's `conformalInference/R/split.R` as the algorithm reference.
[UPSTREAM.md](UPSTREAM.md) records the pinned reference and implementation choices.

For each series, calibrate once on `abs(heldout_y - heldout_predictions)` from
the calibration prefix. With `n` calibration values and miscoverage `alpha`,
the radius is the `ceil((n + 1) * (1 - alpha))`-th smallest score, or infinity
if that rank is `n + 1`. Each test interval is `prediction +/- radius`.
Calibration remains fixed throughout test. No CP training, validation,
normalization, random seed, rolling update, or hyperparameter search is needed.

## Run an experiment

Activate the existing `dscp` environment and run from the repository root:

```bash
python -m sbatch.sbatch_run_split_cp.run_split_cp configs/split_cp_configs/split_cp_lr_air_config.yaml
python -m sbatch.sbatch_run_split_cp.run_split_cp --dataset air --base-predictor lstm --dry-run
python -m sbatch.sbatch_run_split_cp.run_split_cp --dataset solar --base-predictor chronos --num-cores 4
```

Nine presets cover Air, Solar, and Sapflux with LR, LSTM, and Chronos forecasts.
The matching saved prediction artifact must already exist. `--data-path` and
`--output-dir` override the input and output paths. Relative CLI configuration
paths are relative to the current directory; paths inside YAML are relative to
the repository root. `--num-cores` overrides the YAML worker count; parallelism
is across independent series. Results are deterministic across worker counts.
`--dry-run` validates and displays settings without writing files.

Dataset Slurm arrays in `sbatch/sbatch_run_split_cp/` run all three predictors.
They activate `dscp` by default; `SPLIT_CP_ENV` selects another environment.
The NumPy model itself does not depend on PyTorch; the experiment runner uses
the existing OmegaConf, PyTorch reporting helpers, and Matplotlib plots.

## Calibration and test

```yaml
num_cores: 1
data:
  data_path: ./data/air-10_prediction/lr/lr_air-10_data.pkl
  calibration_ratio: 0.66
  test_ratio: 0.34
model:
  prediction_step: 1
  target_quantiles:
    - [0.05, 0.95]
plotting:
  plotting: true
  plotting_seq_len: 200
saving_dir: ./results/split_cp_lr_air/
```

Ratios apply to each complete saved held-out sequence, not the original raw
series. They must be positive and sum to one. Calibration contains
`floor(T * calibration_ratio)` observations; test contains the remainder.
Both partitions must be nonempty. Omitting `test_ratio` uses its complement.
Targets and forecasts may have shape `[T]` or `[T, 1]` and must be finite;
multivariate targets and mismatched lengths are rejected. No `heldout_x` or
strided history windows are required. Only pre-test labels enter calibration.

Presets use 66% calibration / 34% test, following NexCP's allocation. A nominal
80% / 20% split is also available by changing the two ratios. For exact
comparisons, match the saved `target_indices`: QR/SPCI/IQN floor training and
ceil validation separately, while Local-CP floors cumulative boundaries.
These can differ from a single ratio by one observation.

Use an optional zero-based `data.test_start` to specify the boundary exactly.
It overrides the ratio-derived boundary; all preceding rows are calibration.
It can be a single integer for all series or a mapping with exactly the keys
in the prediction artifact:

```yaml
data:
  test_start: {station_a: 667, station_b: 812}
```

For example, with `T=101`, QR's 50%/16% split starts test at index 67,
whereas `calibration_ratio: 0.66` starts it at 66. Set `test_start: 67` to
match QR's test rows. Indices are relative to the shared prediction artifact;
different predictor artifacts may start at different calendar times.

Each `target_quantiles` pair must be symmetric about 0.5; reversed pairs are
accepted and retain their existing tuple log keys. Multiple coverage levels
reuse the same sorted scores. These labels specify nominal total coverage;
the method does not separately control each tail or guarantee coverage
conditional on the current covariates. Settings for online updates or
normalization are rejected instead of silently changing the method.

## Outputs and direct use

Ordinary runs write `resolved_config.yaml`, `log.pkl`, `summary_results.pkl`,
and optional PDF plots. Logs use DSCP's response-scale `lower_interval` and
`upper_interval`, with separate `lower_residual_quantile` and
`upper_residual_quantile` offsets. They include coverage, width, Winkler score,
targets, forecasts, exact target indices, calibration size, quantile rank and
radius, timing, and upstream provenance. Summaries retain the existing
unweighted mean/std across series.

Small calibration sets may require an infinite radius. These intervals cover
every finite target and have infinite width and Winkler score. Summary
standard deviations involving infinity are undefined and stored as `None`;
the intervals are not silently capped. An unbounded band cannot be displayed
as a finite shaded region, although targets and forecasts remain plottable.

```python
from baselines.split_cp import SplitCPResidualIntervalEstimator

calibrator = SplitCPResidualIntervalEstimator().fit(y_cal - predictions_cal)
lower, upper = calibrator.predict_interval(predictions_test, alpha=0.1)
```

Use a separate calibrator per series. The base model and any score choices
must be fixed independently of calibration labels. Base forecasts may use
past observed targets causally, while this conformal radius stays fixed.
For homogeneous one-step comparisons, generate Chronos artifacts with
`prediction_length=1`; older flattened block artifacts mix forecast leads.
The usual exchangeability guarantee does not automatically apply to these
dependent time series, so report empirical coverage alongside width and score.
