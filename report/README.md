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

# Inspect tuning results

From the repository root, display the best five coverage-eligible configurations:

```bash
python report/inspect_tuning_results.py results/tuning/lcp_rnn_lr_air/tuning_results.pkl --top-k 5
```

`--top-k K` ranks eligible entries from `all_trials` by increasing Winkler score,
breaking ties by trial index. It can show more configurations than the tuner
originally included in `top_trials`. If fewer than K trials passed the coverage
filter, it shows all eligible trials. K must be a positive integer.

Without either option, the inspector displays the saved `top_trials`. Use `--all`
to show every trial, including rejected configurations, in trial-index order;
`--all` and `--top-k` cannot be combined. The report includes hyperparameters,
coverage, width, Winkler score, and the path to each trial's resolved YAML.

Individual completed trials can also be inspected while tuning is still running:

```bash
python report/inspect_tuning_results.py results/tuning/lcp_rnn_lr_air/trial_0001/result.pkl
```

`--top-k` requires the combined `tuning_results.pkl`, which is written when the
tuning run finishes.

# Plot saved base predictions

Use `plot_predictions.py` with any base predictor's `*_data.pkl` file from
Sapflux, solar, or air. NumPy and Matplotlib are required.

```bash
python report/plot_predictions.py data/sapflux-solo3-large/lstm/lstm_sapflux-solo3-large_data.pkl --plot-len 1000 --save-dir report/plots/sapflux
python report/plot_predictions.py data/solar_prediction/lr/lr_nsdb-60m_data.pkl --plot-len 1000 --save-dir report/plots/solar
python report/plot_predictions.py data/air-10_prediction/chronos/chronos_air-10_data.pkl --plot-len all --save-dir report/plots/air
```

The script saves one PDF for **every sequence**, plotting `heldout_y` against
`heldout_predictions`. Use `--plot-len all` to plot every held-out observation in
each sequence, or a positive integer to plot the first N steps. Sequences shorter
than N are plotted in full. It accepts lists, one-dimensional arrays, and column
vectors, and checks that targets and predictions have equal lengths. The x-axis
is the zero-based held-out step; timestamps are not stored in these artifacts.
Values are plotted on their saved scale.

The output directory is created automatically. PDF filenames include the input
artifact name, sequence index and identifier, and actual plotted length. Running
the same command again replaces the corresponding PDFs. `--saving-dir` is an
alias for `--save-dir`. Relative paths are resolved from the current directory.
