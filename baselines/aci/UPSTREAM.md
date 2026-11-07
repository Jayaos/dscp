# ACI provenance and implementation choices

Primary reference: Isaac Gibbs and Emmanuel J. Candes,
[Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/pdf/2106.00170),
NeurIPS 2021.

Author reference code:
[ACCode.R](https://github.com/isgibbs/AdaptiveConformal/blob/b7fff0cbd2d3a1e954bc9c426327e355a402d2cd/ACCode.R),
repository commit `b7fff0cbd2d3a1e954bc9c426327e355a402d2cd`.
This is an independent implementation of the mathematical algorithm; no R
implementation was copied. The reference repository did not provide a license
file when inspected.

## Reproduced behavior

- Main simple update, equation (2):
  alpha_next = alpha_current + gamma * (nominal_alpha - miss).
- Gamma 0.005 and nominal miscoverage 0.1, as in section 2.
- The most recent 1250 observed scores, as in the volatility experiment.
- Nominal alpha at the start of evaluation. Historical calibration initializes
  the score pool without first adapting alpha.
- Quantiles use linear interpolation, matching the author's unqualified
  `quantile(...)` calls and R's default type 7. See
  [R quantile documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html).
  The displayed empirical-CDF inverse in the paper is an explicitly selectable
  alternative, `quantile_method: inverted_cdf`.
- No clipping of the adaptive level. The author election routine explicitly
  returns full sets for alpha <= 0 and empty sets for alpha >= 1.
- A score equal to its threshold is covered.

## Dataset adaptation and documented corrections

The default score is absolute residual, explicitly supported in section 1.1
and compared with normalized scores in section 5. The normalized score in the
volatility experiment divides by a positive variance forecast. Our optional
`relative` score reproduces that score/set construction only for strictly
positive point predictions. We do not replace the repository's LR/LSTM/Chronos
point forecasts with the paper's GARCH model.

The paper's window consists of past scores. The R volatility routine has
indexing edge cases (including its first `1:0` slice) and a later slice with
1250 - 1 scores. We use exactly up to 1250 **previously observed** scores and
never include the current score in the interval being evaluated. Boundary
guards follow the author election routine; the GARCH routine lacks them.

Calibration/test partitioning follows this repository's NexCP evaluation:
`floor(T * calibration_ratio)`, default 0.66. Optional calibration-tail tuning
and conditional finite-interval reporting are repository experiment policies.
They do not change ACI's simple update, and the settings/rules are recorded in
the saved output. Full/empty sets remain in coverage and adaptive updates.
