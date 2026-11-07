# Source and implementation fidelity

These baselines independently implement the LQT and SQT+AR algorithms of
Felipe Areces, Christopher Mohri, Tatsunori Hashimoto, and John Duchi,
[Online Conformal Prediction via Online Optimization, ICML 2025](https://proceedings.mlr.press/v267/areces25a.html).

The primary code reference is
[chrismohrii/conformalopt-experiments](https://github.com/chrismohrii/conformalopt-experiments),
pinned to commit
[`940dd5a1e1c2db9ad5402af95b11160fd79bf3d8`](https://github.com/chrismohrii/conformalopt-experiments/commit/940dd5a1e1c2db9ad5402af95b11160fd79bf3d8)
(May 28, 2025). Relevant files:

- [methods.py](https://github.com/chrismohrii/conformalopt-experiments/blob/940dd5a1e1c2db9ad5402af95b11160fd79bf3d8/methods.py):
  `get_linear_qt_predictions` and `qt_on_ar`.
- [main.py](https://github.com/chrismohrii/conformalopt-experiments/blob/940dd5a1e1c2db9ad5402af95b11160fd79bf3d8/main.py):
  grids, candidate ranking, initialization, schedules, and fresh test runs.
- [load_data.py](https://github.com/chrismohrii/conformalopt-experiments/blob/940dd5a1e1c2db9ad5402af95b11160fd79bf3d8/load_data.py):
  score loading and source-specific preprocessing.
- [requirements.txt](https://github.com/chrismohrii/conformalopt-experiments/blob/940dd5a1e1c2db9ad5402af95b11160fd79bf3d8/requirements.txt):
  the original unversioned CVXPY dependency.

The experiment repository has no LICENSE file at this revision. This directory
contains an independent implementation and attribution, rather than a vendored
copy or a claim that the experiment code carries an MIT license.

## LQT

Using zero-based time `t`, the feature vector contains the preceding `p`
scores in chronological order followed by the configured constant feature
`bias`. The threshold is the inner product of this vector and the current
coefficient vector. After observing the score, the coefficient update adds
`eta_t * (miss_t - alpha) * feature_t`, with
`miss_t = 1[score_t > threshold_t]`.

For `p > 0`, every initial coefficient is `1/p`; the first `p` thresholds
are zero and those steps do not update coefficients. For `p = 0`, the
coefficient starts at `initial_quantile`, whose default is one. There is no
projection, regularization, or threshold clipping. The lag order can be zero:
this is a constant-feature member of the LQT family.

## SQT+AR

The threshold is an AR quantile prediction plus a scalar correction. The AR
fit minimizes summed pinball loss at `1-alpha`, using lag scores and an
intercept feature of one. The correction starts at zero and adds
`eta_t * (miss_t - alpha)` after each active prediction. Both components use
only observations available before the next prediction.

The experimental code has specific startup and window behavior:

- The first `p` thresholds are zero, with no correction updates.
- At the first active step, the regression includes one all-zero feature row.
  For `p > 0`, its target is the preceding, already-observed score.
- At later steps, one previous feature/score pair is appended. The source
  trimming condition retains up to `ar_window_size + 1` rows. This behavior
  is preserved, including the nominal default window of 200.
- The regression has no coefficient constraints or regularization. The
  publication implementation calls CVXPY at every active step.

There is one necessary causal repair for `p = 0`. At the first step, upstream
pairs its zero feature row with `scores[-1]`, which is the final outcome in
the supplied stream. Because the feature row is zero, its loss is constant
with respect to all regression coefficients. Here its target is zero and the
initial scorecast is zero, avoiding access to a future outcome while preserving
the mathematical minimizers. Floating-point solver paths need not be identical
after removal of that constant. For a one-point stream, the upstream negative
index aliases the current feature row and directly leaks the current outcome;
the explicit zero row also repairs that corner case.

The default initialization thresholds of zero are singleton prediction sets
for absolute-residual scores. Upstream comments describe them as empty, but
the actual threshold values and inclusive coverage comparisons define
singletons. This implementation follows those values.

## Hyperparameters and state

The default grids reproduce `main.py`:

| Parameter | Values |
| --- | --- |
| Learning rate | `1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000` |
| AR order, both methods | `0, 1, 2` |
| LQT bias feature | `0.1, 1, 5, 10, 200, 100, 1000` |

Candidates are ordered by learning rate, lag order, then LQT bias. The bias
order deliberately preserves the source code's 200-before-100 order; the
paper's table lists those values in ascending order. Exact loss ties keep the
first candidate.

The experiment code selects the least mean pinball loss among candidates with
`abs(coverage - (1-alpha)) <= 0.01`, falling back to the unrestricted minimum
if no candidate qualifies. This **two-sided** condition differs from the
paper's Appendix B.1 prose, which describes only a lower coverage bound.
The implementation uses the code's condition and exposes its tolerance.

The paper evaluates both fixed and decaying learning rates. The presets use
fixed rates; configuring `decaying` gives `eta_t = lr * (t+1)^(-0.6)` by
default. The time index includes the initial zero-threshold steps.

Each candidate runs from scratch on calibration. The chosen method then runs
from scratch on test, resetting coefficients, regression history, correction,
and schedule time. No calibration state is transferred to testing. Selection
and state are separate for every saved sequence and coverage level.

## Adaptation to this repository

The statistical update rules above remain independent of DSCP. Integration
changes are limited to the experimental setup and output handling:

- Saved point forecasts supply raw absolute-residual scores. They are not
  centered or normalized. Prediction sets use the corresponding symmetric
  interval, including empty sets when thresholds are negative.
- NexCP's chronological boundary `floor(T * calibration_ratio)` replaces the
  upstream dataset-specific split ratios. Calibration alone selects all
  hyperparameters; test outcomes are revealed sequentially for online updates.
- The source loader's unconditional removal of the first 30 scores is a
  dataset-cleaning convention and is not applied to these forecast artifacts.
  ERCOT-specific initialization and its preregistered window of 10 are also
  not used for Air, Solar, or Sapflux.
- Regression scorecasts can be cached across calibration candidates that differ
  only in learning rate, because neither the regression data nor its objective
  depends on that rate. States and cached paths are never shared across
  independent sequences or different coverage levels.
- Coverage includes all test timestamps. Conditional width and Winkler means
  exclude full/empty sets, whose counts and rates are reported separately.
  These are the benchmark's reporting conventions, not the authors' original
  score-threshold `set_size` metric.
- Solver failures and nonfinite numerical states are surfaced as errors.
  They are not silently clipped or reclassified as algorithmic full sets.

## Solver and environment

The original experiments use `cvxpy` without a version pin and call the default
solver without recording its version. Consequently, reproducing their
historical numerical solver behavior exactly is not possible from the source
alone, especially when quantile regression has multiple minimizers.

This repository pins CVXPY 1.6.5 and Clarabel 0.11.1 in the existing
[env-dscp.yml](../../envs/env-dscp.yml), and defaults explicitly to
`solver: CLARABEL`. These dependencies support the shared Python 3.9
environment. Solver name and options are configurable and saved. This is an
explicit reproducibility choice for the same regression objective, not a claim
about the authors' historical solver installation.

## Why the generic package is not the reference

The authors also provide
[`conformalopt`](https://github.com/chrismohrii/conformalopt/tree/86bbd61408ed3602269e39754d7b3db73948a240),
examined at commit `86bbd61408ed3602269e39754d7b3db73948a240`.
Its SQT+AR path differs from the experimental implementation: the scalar
component starts at one, the scorecaster waits for ten observations, quantile
regression uses Statsmodels, and the default AR search contains only order one.
Its tuning selection also tests the last candidate's coverage flag when
choosing between constrained and unrestricted optima.

Those package details are not used here. Pinning the experimental functions
avoids mixing the publication's experiment defaults with the later reusable
package interface.
