# DistMatch provenance and compatibility

This adapter follows the executable DistMatch baseline in
[enver1323/dist_match_conformal](https://github.com/enver1323/dist_match_conformal),
commit `d9fd84dde4a2b92cf5340a3cae578bd0fe489ccd`, cloned at
`dist_match_conformal/` in this workspace. The method is described in
[DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential
Conformal Prediction](https://openreview.net/pdf?id=SxBuTatzGe).

The implementation was adapted from these source files:

- `code/models/uncertainty/dist_match/tree.py`: greedy anchor selection,
  replacement bootstrap, ancestor-augmented leaves, leaf QRFs, beta selection,
  and averaging tree interval endpoints.
- `code/models/uncertainty/dist_match/dist_match.py`: signed residual windows,
  calibration alignment, and sequential interval construction.
- `code/models/uncertainty/dist_match/utils.py`: two-sample KS statistic.
- `configuration/config/model_uc/dist_match.yaml`: experiment defaults.

## Behavior retained from the official code

Calibration residuals are signed `y - point_prediction`. For window length W,
training pair i is `(residuals[i:i+W], residuals[i+W])`. The initial partition
therefore has `calibration_size - W` candidate pairs. Each outer tree samples
`floor(candidate_pairs * bagging_ratio)` indices with replacement.

Matching uses the strict comparison `KS(window, anchor) < gamma`. At each node,
anchors are scanned in subset order; the first candidate with the largest
matched population wins. Both child counts are checked against the minimum
node size before removing the anchor. The selected anchor is removed from the
matched child. Terminal leaves receive their subset followed by every ancestor
anchor in immediate-parent-to-root order, including bootstrap multiplicity.
The partition remains fixed during sequential prediction.

At a queried leaf, a fresh `sklearn_quantile.RandomForestQuantileRegressor`
fits flattened windows and residual targets in the selected normalization
units. Defaults match the official
baseline: W=100, gamma=0.1, ten outer trees, bootstrap ratio 0.9, minimum node
size zero, ten beta candidates, and ten QRF estimators of maximum depth two.
For each requested mass, beta ranges from zero to alpha inclusively. Each tree
selects its narrowest candidate interval, then lower and upper endpoints are
averaged separately across trees. Leaf memories grow without a recency cap.

The paper describes a non-strict KS comparison and does not specify the same
ancestor-augmentation bookkeeping. This adapter preserves those executable
code details; it does not claim that the code and paper are interchangeable.

## Deliberate integration changes

- DSCP supplies saved point forecasts and a chronological calibration prefix.
  This adapter does not train or modify the underlying forecasting model.
- Every newly observed residual is inserted exactly once with its
  pre-observation window. The official wrapper inserts the final calibration
  pair again before its first test prediction; this duplicate is omitted.
- All requested coverage levels use the same per-series state. Predicting
  additional quantile pairs does not append observations or alter the trees.
- Randomness is local and reproducible: independent bootstrap streams per tree,
  and QRF seeds determined by the run seed, tree index, and observed-update
  count. QRFs use one worker. Neither global NumPy RNG state nor query ordering
  changes results. This intentionally differs from upstream's global RNG and
  the discarded QRF fit inside its update method, so numeric equality with an
  entire upstream experiment is not promised.
- All requested quantile levels are evaluated with one leaf QRF fit per tree
  per `predict_intervals` call. Extra calls are deterministic and do not advance
  state. `observe` only appends a pair; the next prediction refits the leaf QRF.
- `data.normalize_residual` is the sole normalization setting and defaults to
  `true`. All LR, LSTM, and Chronos presets enable it. Setting it to `false`
  uses raw residuals for both input windows and QRF targets. The former
  `data.normalize` and `data.normalization_mode` keys are rejected with a
  migration error, and input-only residual standardization has been removed.
- With residual normalization enabled, target statistics use the held-out
  prefix before the active evaluation split: `heldout_y[:train_end]` for
  validation or `heldout_y[:validation_end]` for final test. Saved `train_y` is
  prepended when present. Artifacts without it, including existing Chronos
  artifacts, use only that held-out prefix; unavailable pre-forecast context
  is omitted. A supplied `train_y` is still validated, and invalid history is
  rejected. At least two historical targets are required. The target mean and
  sample standard deviation (`ddof=1`) are frozen for that run; constant target history
  uses scale one. Both residual windows and QRF targets are divided by the
  target standard deviation. The mean cancels when subtracting a standardized
  prediction from a standardized target. Predicted residual quantiles are
  multiplied by the same scale before adding them to the original saved
  forecasts, keeping all intervals and metrics in original units. Per-series
  metadata records the statistics' source, mean, standard deviation, sample
  count, and held-out cutoff. The source explicitly distinguishes combined
  history from the held-out-prefix fallback.
- This residual scaling follows the original normalization formula while retaining
  DSCP's saved forecasts, alignment, and chronological splits. It does not
  retrain the base predictor on the original standardized features and targets,
  so it does not reproduce upstream's full forecasting experiment. Nonfinite
  inputs are rejected instead of silently replaced with zero. The estimator
  core uses the supplied residual units without a normalization option; data
  preparation and the runner handle scaling and restoration to original units.
- Two-sample KS uses sorted equal-length windows and exact integer empirical
  CDF counts, including ties. Computation is blocked, and the pairwise cache is
  boolean. No quadratic floating-point distance matrix or tree-owned mask is
  retained. The initial cache still requires quadratic disk or memory space,
  and matching still requires quadratic pair comparisons.
- With `cache_dir`, a fingerprint of sorted input windows,
  window size, threshold, and cache algorithm version names a persistent `.npy`
  boolean memmap. Publication is atomic. Without a directory, fitting uses an
  in-memory boolean matrix within the memory budget, otherwise a temporary
  memmap removed after fitting. All matrix mappings close after the trees fit.
  The configured memory budget limits the pairwise cache allocation; patches,
  integer rank arrays, QRFs, and growing leaf state also require memory.
- The estimator core only needs the NumPy/SciPy/scikit-learn/sklearn-quantile
  dependency path. sklearn-quantile is imported lazily. The core does not use
  Hydra, Darts, wandb, XGBoost, Torch, upstream pickle filenames, or upstream
  multiprocessing. The DSCP experiment runner uses the project's Torch-based
  metrics and artifact utilities.
- The validated sklearn-quantile 0.1.1 environment can return nonfinite
  quantiles from a single-estimator QRF on nonconstant leaves. The adapter
  requires `qrf_n_estimators >= 2`; the official default of ten is unaffected.
- That dependency also accumulates its conditional CDF in float32. If the
  final sum remains below one, its q=1 output can remain an unfilled zero even
  when all supported targets are positive. For requested q=0 and q=1 only, the
  adapter uses the exact minimum and maximum targets with positive weight in
  the fitted forest's queried leaves. It preserves all interior quantiles
  exactly as returned; it does not sort, rearrange, or clip their values.
- Interior quantiles can also cross because the dependency accepts a CDF up
  to `1e-6` below the requested quantile and then extrapolates with a linear
  interpolation fraction greater than one. The estimator raises a diagnostic
  `DistMatchCrossedBoundsError` for the selected negative-width candidate.
  Final test evaluation records and excludes that entire timestamp across
  coverage levels, then observes its residual exactly once. It does not repair
  or reorder quantiles. Validation tuning continues to raise on this failure.
  Saved metrics, plots, and rolling windows use only successful predictions;
  exclusions and their counts are explicit. This is a DSCP evaluation policy,
  not an upstream algorithm feature or a guarantee of coverage on the full
  test set. Other numerical and programming errors remain fatal.

`use_beta_search=False` is a labeled ablation that uses the requested lower and
upper quantiles directly. Other constructor changes, normalization, and the
calibration prefix size should be recorded with each run. `memory_size` counts
unique chronological window-target pairs; a tree's leaf membership can include
bootstrap duplicates and ancestor copies.

## Validation boundary

`tests/test_distmatch_model.py` compares the blocked KS implementation against
SciPy with repeated values and exact threshold boundaries. An independent
direct-SciPy tree implementation verifies split anchors and ordered leaf
membership. A QRF fixture compares residual endpoints using those independent
leaf memberships. When the official clone is present, another fixture executes
its original tree implementation with only the unused XGBoost import removed
and checks its partition and ordered leaf membership against the adapter.
Streaming checks cover window/target alignment, one update
per tree, fixed topology, causal prefixes, query-order independence, local RNG
reproducibility, normalization, and boolean-cache backend parity.

These are implementation checks. They do not establish empirical coverage on
DSCP datasets or reproduce the paper's forecasting models, dataset splits,
hyperparameter selection, software environment, or aggregate reported scores.
