This run checks ACI, LQT, and COP on the COP authors' cached AMZN/Prophet
forecasts against [Table 2 of Hu et al., arXiv:2512.07770v2](https://arxiv.org/html/2512.07770v2#S4.T2).
It uses the authors' experiment functions in `upstream_methods.py`, separately
from this repository's general benchmark implementations. It runs in the
existing `dscp` environment from `envs/env-dscp.yml`; Prophet does not need to
be fitted again.

From the repository root, submit one job for all three methods:

```bash
sbatch sbatch/sbatch_reproduce_cop_table2/run_amzn_prophet.sbatch
```

To inspect the setup or run locally in the same environment:

```bash
python -B sbatch/sbatch_reproduce_cop_table2/run_amzn_prophet.py --dry-run
python -B sbatch/sbatch_reproduce_cop_table2/run_amzn_prophet.py
python -B sbatch/sbatch_reproduce_cop_table2/run_amzn_prophet.py --method aci
```

`--method` accepts `all`, `aci`, `lqt`, or `cop`. Results are printed only to
standard output, captured by the Slurm run log. The runner creates no result
files, plots, caches, or checkpoints and makes no network requests. `-B`
prevents Python bytecode cache writes.

The two required local inputs are:

| File | Contents and provenance |
| --- | --- |
| `data/amzn_prophet.npz` | The authors' cached `forecasts` array, with 3020 observations. Its verified Git blob SHA is `1a1b38e0d9abb355e06936edfd4dea5e509f83d6`. |
| `data/amzn_prophet_targets.csv` | Columns `timestamp,Open`, extracted from the authors' `djia.csv` on the full union of stock trading dates. There are 3020 rows, from 2006-01-03 through 2017-12-29. The missing AMZN opening price on 2010-04-01 is left blank and interpolated by the loader, matching the authors' dataset loader. |

The NPZ contains forecasts, so the target CSV is also required. These are input
datasets, not generated experiment results. `data/` is ignored by Git: copy
both inputs to the cluster's `data/` directory with the repository.

All upstream references are pinned to commit
`a41c0dcadc10979d8521ab9b5b8f3d460807d883`:
[forecasts](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/datasets/proc/AMZN_test/prophet.npz),
[raw stock data](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/datasets/djia.csv),
[dataset loader](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/datasets.py),
[AMZN configuration](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/configs/AMZN_test.yaml),
and [method implementations](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/core/methods.py).
The copied functions retain their MIT license in `UPSTREAM_LICENSE`.

The run follows the authors' sequential protocol, with no NexCP calibration/test
split. Each candidate starts with independent state and processes all 3020
observations in order. Overall miscoverage is 0.1. The lower and upper score
streams are `forecast - target` and `target - forecast`; each uses miscoverage
0.05. Forecast horizon is one step. The authors' interval wrapper sorts reversed
endpoints with `min`/`max`; this behavior is preserved.

| Method | Default configuration | Learning-rate candidates |
| --- | --- | --- |
| ACI | Burn-in 100; history window 10,000,000; the authors' quantile and infinite-set handling | 0.1, 0.05, 0.01, 0.005 |
| LQT | Autoregressive order 2; bias feature 5; fixed learning rate; coefficients initialized using the first 100 scores | 10, 5, 1, 0.5, 0.1, 0.05, 0.01 |
| COP | Empirical CDF; window 100; correction scale 0.5; learning rate scaled by the recent score range | 1, 0.5, 0.1, 0.05 |

The rate candidates follow Appendix D. The released AMZN YAML lists only
`[10, 1, 0.1, 0.01]` for LQT; this run also evaluates the three additional rates
listed in the paper. LQT's order and bias come from the AMZN YAML. The authors
mention a hyperparameter search without recording every choice used for Table 2,
so these YAML settings cannot establish the exact configuration behind that row.

The primary log metrics mirror the released executable scripts:

- Coverage is the fraction of targets inside the final closed intervals after
  zero-based index 100, i.e. `intervals[101:]`, giving 2919 evaluated points.
- Mean and median width use all 3020 intervals, as printed by the authors'
  `base_test.py`. Infinite intervals remain in these summaries.
- Supplemental mean and median widths over `intervals[101:]` make the effect of
  the evaluation slice explicit.

These differing slices come from
[base_plots.py](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/base_plots.py)
and [base_test.py](https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/a41c0dcadc10979d8521ab9b5b8f3d460807d883/tests/base_test.py).
The repository does not provide an executable Table 2 aggregation script.

For comparison, the log prints the published AMZN/Prophet values and differences
for every candidate:

| Method | Coverage (%) | Mean width | Median width |
| --- | ---: | ---: | ---: |
| ACI | 90.2 | infinity | 46.97 |
| LQT | 89.3 | 31.42 | 15.26 |
| COP | 89.6 | 39.86 | 27.91 |

The log identifies agreement at the paper's displayed precision for each
candidate and contains all measured values. Reproduction measurements are not
written to this document or to result files.

No automatic best-candidate selector is imposed. The paper does not identify the
selected rates or a precise selection rule, so matching Table 2 exactly is not
guaranteed. The released code also differs from the prose in two details: the
stock input extends through 2017 although Section 4.3 says 2014, and the COP
update uses the preceding 100 scores excluding the just-observed score. This run
preserves the supplied data and executable update conventions for reproducibility.
