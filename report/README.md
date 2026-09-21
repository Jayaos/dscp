# Inspect saved runs

From the repository root, use the Python environment used for the experiment:

```bash
python report/inspect_run_results.py results/rescp/air/lstm/run --rolling-window-size 100
```

Pass one method run directory containing `log.pkl`. The report displays method
identity, dataset, base predictor, saved hyperparameters, and a separate table
for each quantile pair with mean and standard deviation of:

- Coverage
- Interval width
- Winkler score
- Delta coverage
- Rolling coverage
- Delta rolling coverage
- Rolling undercoverage

The report reads `resolved_config.yaml` when present. Older runs without that
file can still be summarized, but missing identity/configuration details are
shown as `n/a`. Method identity inferred from saved model settings or result
paths is labeled accordingly. Saved IQN model configurations are also available
from `log.pkl` when the YAML is missing.

Each metric is averaged within each sequence, then those sequence means receive
equal weight, even when sequences have different lengths. Standard deviations
use `ddof=0`, matching the existing run summaries. The stored per-timestep width
and Winkler scores retain the method's scoring and scale conventions.

Rolling coverage uses all complete overlapping windows of the requested size,
with stride one and no padding. Window coverages are averaged within each
sequence before computing mean/std across sequences. Sequences shorter than the
window are excluded only from rolling metrics; the report shows the number used.
If no sequence is long enough, rolling metrics are `n/a`.

For rolling undercoverage, compute each sequence's mean positive shortfall:

```text
u_j = (1 / N_j) * sum_t max(target_coverage - rolling_coverage[j, t], 0)
MRU = (1 / M) * sum_j u_j
std = sqrt((1 / M) * sum_j (u_j - MRU)^2)
```

Here `N_j` is the number of complete windows for sequence `j`, and `M` is the
number of sequences with at least one complete window (the displayed rolling
sequence count). The shortfall is clipped at zero **before** averaging windows,
so overcoverage in one window does not cancel undercoverage in another. Each
sequence has equal weight regardless of its number of windows. The returned
summary keys are `avg_rolling_undercoverage_mean` and
`avg_rolling_undercoverage_std`.

Both deltas are signed: coverage minus nominal target coverage, where the target
is the upper quantile minus the lower quantile. Widths/scores may be infinite
for small SplitCP calibration sets; their means remain infinite and undefined
standard deviations display as `n/a`.

The same report is callable from Python:

```python
from report.inspect_run_results import inspect_results

summary = inspect_results("results/rescp/air/lstm/run", rolling_window_size=100)
```
