# COP source and implementation fidelity

This baseline implements Conformal Optimistic Prediction from Dongjian Hu,
Junxi Wu, Shu-Tao Xia, and Changliang Zou,
[Distribution-informed Online Conformal Prediction, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5f8241131e6fd428aa49914da76b8ad0-Abstract-Conference.html).
The [paper's HTML version](https://arxiv.org/html/2512.07770v2) provides Algorithm 1,
the experimental setup in Section 4.1, and learning-rate details in Appendix D.

The official implementation is
[creator-xi/Conformal-Optimistic-Prediction](https://github.com/creator-xi/Conformal-Optimistic-Prediction),
pinned to commit
[`a41c0dcadc10979d8521ab9b5b8f3d460807d883`](https://github.com/creator-xi/Conformal-Optimistic-Prediction/commit/a41c0dcadc10979d8521ab9b5b8f3d460807d883)
(December 8, 2025). Relevant files:

- [core/methods.py](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/core/methods.py):
  the `COP` function and its active ECDF update.
- [tests/base_test.py](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/base_test.py):
  score conventions, two-tail construction, parameter sweeps, and endpoint
  postprocessing.
- [tests/configs](https://github.com/creator-xi/Conformal-Optimistic-Prediction/tree/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/configs):
  the published experiment grids and dataset-specific windows.
- [tests/base_plots.py](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/base_plots.py):
  evaluation after `T_burnin + 1`.
- [LICENSE](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/LICENSE):
  MIT, copyright 2025 Junxi Wu. The notice is retained in
  [UPSTREAM_LICENSE](UPSTREAM_LICENSE).

## Active update and timing

Write `a` for a scalar stream's nominal miscoverage, `s[t]` for its observed
score, `h[t]` for the primary threshold, and `q[t]` for the issued threshold.
The default initial values are `h[0] = q[0] = 0`.

At zero-based time `t`, the method first issues the set defined by `q[t]`.
After the outcome arrives, its gradient is

```text
g[t] = a                      if s[t] <= q[t]
       -(1 - a)               otherwise
h[t+1] = h[t] - eta[t] * g[t]
```

The adaptive rate at `t > 0` uses the range of
`s[max(0, t-window_size):t]`; at time zero it uses the base learning rate.
The current score is excluded from this range. When `proportional_lr` is false,
the rate is the configured base rate at every step.

For `t < window_size`, the distribution correction is zero. Otherwise,

```text
F = mean(s[t-window_size:t] <= h[t+1])
q[t+1] = h[t+1] - eta[t] * scale * (F - (1 - a))
```

This preserves an important source-code indexing detail: the current score
updates the primary threshold, but the ECDF window used to refine the next
threshold still excludes that current score. The paper's mathematical
description uses a CDF based on the available score history; the released code
uses this one-score-stale window. No current or future target enters an interval
before that interval is issued.

Equality with a threshold is covered, and ECDF comparisons are inclusive.
There is no threshold clipping or coefficient projection.

The streaming implementation consumes the final available outcome so that its
state can continue across the calibration/test boundary. The upstream array
routine does not store a next threshold after the last array element; consuming
that final outcome does not change any threshold already issued.

The source's `T_burnin` parameter serves three roles: the adaptive-rate window,
the ECDF window, and the initial period without refinement. This baseline keeps
those roles coupled under the clearer name `window_size`.

The upstream function calls `np.random.seed(42)`, but the active ECDF path is
deterministic. Resetting the caller's global random generator is unnecessary
for reproducing its output and is not part of this baseline. Commented-out noisy
CDF and Gaussian KDE experiments are not enabled.

## Publication settings

Section 4.1 uses 90% nominal interval coverage, ECDF, scale factor 0.5, and a
window of 100. Appendix D and the official COP configurations use base-rate
candidates `[1, 0.5, 0.1, 0.05]`.

The real-data configurations use signed scores in two independent streams:
`prediction - y` for the lower endpoint and `y - prediction` for the upper
endpoint, each with half the interval miscoverage. The same hyperparameter
candidate is used for both tails. These produce potentially asymmetric
intervals even when the nominal tail probabilities are equal.

The official windows vary by experiment: 100 for stock datasets, 300 for
electricity demand, and 500 for climate temperature. The repository presets
use the paper's general window 100 rather than inventing mappings from those
datasets to Air, Solar, and Sapflux. Window, scale, and learning rate are
configurable.

The optional absolute-score mode is a score-function adaptation supported by
the paper's general framework and its experiment runner. It is not the default
score convention of the published real-data experiments.

## Calibration and test integration

The authors' runner sweeps all learning rates on one uninterrupted stream. It
does not implement a calibration/test split or an automatic selector; Appendix D
describes choosing the best-performing candidate without specifying an
executable tie-breaking or coverage-constrained rule.

This repository introduces the requested NexCP split and restricts selection
to calibration. The default candidate ranking excludes the first
`max(window_size_grid) + 1` calibration timestamps. The single default window
of 100 reproduces the original plotting cutoff of 101; when windows are tuned,
this common cutoff compares candidates on identical target timestamps. Those
observations still initialize and update candidate state. The cutoff can be
configured independently through `tuning.warmup_steps`.

Parameters are selected independently for each sequence and requested interval
coverage. Among candidates whose two-sided calibration interval coverage is at
least nominal coverage minus the configured tolerance, select the lowest mean
pinball loss. Signed-score loss adds the two tail losses. If no candidate
satisfies the coverage floor, select the lowest loss across the grid.
**This calibration selection rule is a benchmark protocol, not an algorithm
prescribed by Hu et al.**

After selection, the default `calibration_replay` protocol restarts the chosen
configuration, replays calibration, and continues into the complete test suffix.
This preserves the authors' uninterrupted update sequence while adding a clean
hyperparameter-selection boundary. Hyperparameters stay fixed in test; online
state continues to update after each observed target.

The optional `cold_start` protocol starts over at test. It is explicitly
recorded in the configuration. Every test timestamp is evaluated under either
protocol; the original plotting cutoff is not applied again to discard test
predictions.

The base forecasts are supplied by existing repository artifacts. Upstream
forecast-model fitting, log transformations for particular datasets, and model
warmup are not rerun or transferred to those artifacts.

## Endpoint and metric conventions

The authors' `tests/base_test.py` wrapper uses a default minimum size of zero
and reorders crossing endpoints through `minimum` and `maximum` operations.
That postprocessing can turn a mathematically empty set into a nonempty
interval. Here crossing endpoints are represented as empty sets, preserving
the repository's agreed exceptional-set policy. This is a deliberate wrapper
difference; the COP scalar updates are unchanged.

A negative signed threshold alone is not an empty set. With signed tails, the
interval is empty only when `prediction - q_lower > prediction + q_upper`.
With absolute scores, a negative threshold is empty. A zero-width interval
remains a singleton.

Coverage and online feedback retain all test timestamps. Conditional width and
Winkler summaries exclude full and empty sets, whose counts and rates are
reported separately. Full-test diagnostic conventions match the existing ACI
reporting utilities. These are repository reporting choices rather than the
publication's average/median-width table format.

Numerical failures are not converted into algorithmic full sets. COP's
long-term guarantees do not imply a per-timestamp conditional coverage
guarantee, and this implementation does not claim one.

## Environment

The active COP ECDF recurrence uses NumPy. Existing repository dependencies
cover configuration, plotting, and reporting, so
[env-dscp.yml](../../envs/env-dscp.yml) needs no COP-specific additions or
replacement.

The upstream `darts==0.30.0` requirement belongs to its broader experimental
forecasting stack. It is unnecessary for applying COP to already-saved
forecasts, and the shared environment's forecasting package versions are not
changed.
