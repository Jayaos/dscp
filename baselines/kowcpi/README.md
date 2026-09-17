# KOWCPI baseline

This baseline adapts the residual interval estimator in
[`KOWCPI_Codes`](../../KOWCPI_Codes/) to saved point forecasts. See the
[Slurm instructions](../../sbatch/sbatch_run_kowcpi/README.md) for running it.

## Data splits and tuning

The normal YAML defines the outer split of the saved heldout point forecasts:

```yaml
data:
  calibration_ratio: 0.66 # Test uses the remaining 1 - calibration_ratio.
```

`calibration_ratio` must be strictly between zero and one. The initial
calibration prefix contains `floor(N * calibration_ratio)` observations; the
remaining suffix is the final test period, corresponding to
`1 - calibration_ratio`. The point predictor's original training observations
are not used by this baseline.

Only the tuning YAML reserves a validation period:

```yaml
tuning:
  model_selection_valid_ratio: 0.15
```

This is a fraction of the calibration prefix, not the full saved sequence.
For 1,000 saved forecasts, tuning initializes on indices `[0, 561)`, evaluates
`[561, 660)`, and never processes the final test values at `[660, 1000)`.
The split is fixed across window-size candidates. Normalization and initial
AIC bandwidth selection use only the earlier history. With online updates
enabled, each validation residual enters history after its interval is issued.
Candidates pass the configured coverage filter and are ranked by Winkler score.
Saved tuning results identify the evaluation split and its boundaries.

A final normal run with the selected model settings initializes on the full
calibration prefix `[0, 660)` and evaluates `[660, 1000)`. It refits normalization
on that full prefix and, when `bandwidth: null`, selects bandwidth there by AIC.
The tuning ratio has no effect on this final run. The Slurm arrays still use
their configured template; apply the selected settings or pass a saved selected
configuration to the config-file launcher before final evaluation.

The separation of tuning and final evaluation follows the paper's experimental
design. This project retains its coverage-then-Winkler selection rule; the paper
selects the smallest average width meeting validation coverage. The data ratios
here apply to heldout forecasts, not the paper's full-dataset 7:1:2 split.

The old `data.train_ratio` and `data.valid_ratio` keys are rejected with migration
guidance. Replace them with their sum as `calibration_ratio`, then add the
tuning-only setting above. An existing `data.test_ratio` is accepted for backward
compatibility only when it equals `1 - calibration_ratio`; it is unnecessary in
new configs. Consolidating the old
`floor(N * train_ratio) + ceil(N * valid_ratio)` boundary can shift it by one
observation. Normalization now uses the full calibration prefix for final runs,
rather than the old `train_ratio` sub-prefix.

## Bandwidth selection

The reference is the active `fit()` path in
[`KOWCPI_Codes/weighted_nw.py`](../../KOWCPI_Codes/weighted_nw.py).
For each candidate bandwidth, it constructs the kernel matrix, retains its
diagonal, and normalizes each row to obtain `W`. The selection criterion is

```text
RSS = sum((y - W @ y)**2)
trace_ss = sum(W**2)
AIC = log(RSS) + (n + trace_ss) / (n - trace_ss - 2)
```

This selector uses ordinary normalized kernel weights. The reference's RNW
`compute_rss()` and `compute_smoothing_matrix()` helpers are not called by
`fit()`. Empirical-likelihood weight correction is applied when predicting
quantiles.

Like the reference, grids with at most 12 candidates are searched directly.
Larger grids use eight evenly spaced grid indices, followed by five candidates
within 70% to 130% of the winning bandwidth, clipped to the original grid bounds.
The refined candidates need not belong to the original grid. The selected
bandwidth is reused for subsequent online predictions.

Set `model.bandwidth: null` to enable selection. In the runner,
`model.bandwidth_range: [1.0, 10.0, 100]` specifies `linspace(1, 10, 100)`;
setting a numeric `model.bandwidth` skips selection.

Positive RSS values use the source's unmodified logarithm, including values
below `1e-12`. For degenerate cases, this port keeps numerical safeguards:
zero RSS uses the smallest positive normal float inside the logarithm, and
the AIC denominator is bounded below by `1e-8`. These cases can differ from
the reference's unguarded formula. The stable scalar solver used for prediction
also retains the earlier fix for SciPy finite-difference warnings.

The bandwidth regression tests compare against the local reference when
`KOWCPI_Codes/weighted_nw.py` is available. Those comparisons are skipped in
checkouts without that ignored directory; the standalone regression tests
still run.
