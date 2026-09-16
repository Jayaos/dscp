# SplitCP algorithm reference

Reference: Ryan Tibshirani's `conformalInference`, function
`conformal.pred.split` in
[`conformalInference/R/split.R`](https://github.com/ryantibs/conformal/blob/15c51c66e3e5cab578a5c4ebd7494685efa83788/conformalInference/R/split.R).

- Reference repository: https://github.com/ryantibs/conformal
- Inspected commit: `15c51c66e3e5cab578a5c4ebd7494685efa83788`
- Reference file Git blob: `cb5f1f8981f76b0a5f6dca1386f9fe98c2c11178`
- Reference repository license: GPL-2.0 (see its
  [LICENSE](https://github.com/ryantibs/conformal/blob/15c51c66e3e5cab578a5c4ebd7494685efa83788/LICENSE)).
- Associated paper: Lei, G'Sell, Rinaldo, Tibshirani, and Wasserman (2018),
  *Distribution-Free Predictive Inference for Regression*.

The local code is an independent Python implementation of the mathematical
unweighted, unscaled split-conformal procedure. No R source is vendored or
translated line by line. The reference documents the algorithm and supplies
the comparison semantics; it is not an R runtime dependency.

The relevant reference behavior is: compute absolute calibration residuals,
augment their empirical distribution with an infinite score carrying one
observation's weight, and take the inverse-CDF quantile at `1 - alpha`.
With equal weights this is rank `ceil((n + 1) * (1 - alpha))`, counting from
one. The radius is infinite when that rank is `n + 1`. The radius is added to
and subtracted from each point forecast and calibration stays fixed.

DSCP-specific choices:

- Base-model fitting has already happened before artifact generation. Only
  saved held-out targets and point predictions are required here.
- The held-out suffix is split chronologically into calibration and test,
  independently for each series, using every observation. Upstream's random
  fitting/calibration split and model-fitting callbacks are unnecessary.
- All pre-test observations supply calibration; optional exact test-start
  indices allow comparisons on precisely the same artifact rows.
- Scores use original response units, equal weights, and no fitted scale.
  Weighted covariate-shift intervals, local MAD scaling, and online updates
  are outside this baseline.
- Decimal probability arithmetic avoids binary roundoff adding an unwanted
  rank at integer boundaries. No interpolation or clipping to the largest
  finite residual is used. Floating-point bitwise parity with R is not claimed.
- Symmetric `target_quantiles` pairs encode nominal coverage for DSCP's log
  schema; they do not assert equal actual error rates in the two tails.
- Infinite-radius intervals retain infinite width and Winkler score; summary
  standard deviations involving infinity are recorded as `None` (undefined).

Chronological dependent forecast errors need not be exchangeable. The ordinary
split-conformal marginal coverage theorem is therefore not an automatic
guarantee for these time-series experiments; evaluate coverage empirically.
