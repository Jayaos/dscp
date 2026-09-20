# SPCI Slurm jobs

Submit one array per dataset from the repository root on the configured cluster:

```bash
sbatch sbatch/sbatch_run_spci/run_spci_air.sbatch
sbatch sbatch/sbatch_run_spci/run_spci_solar.sbatch
sbatch sbatch/sbatch_run_spci/run_spci_sapflux.sbatch
```

The scripts follow [`../sbatch_run_hopcpt/`](../sbatch_run_hopcpt/): the same
account, Inferno QoS, four CPUs, 16 GB host memory, four-hour time limit,
repository path, and notification address. SPCI fits quantile random forests
on CPUs, so these jobs do not request a GPU. They activate the `spci` conda
environment defined in [`../../envs/env-spci.yml`](../../envs/env-spci.yml).
`LOKY_MAX_CPU_COUNT` caps the runner's `n_jobs=-1` at `SLURM_CPUS_PER_TASK`,
and BLAS/OpenMP threads are limited to one to avoid nested parallelism.
Adjust resources with `sbatch --cpus-per-task=N --mem=SIZE --time=HH:MM:SS`
as needed; runtime and memory depend on sequence count and length.

## Array tasks

| Task ID | Base predictor |
| --- | --- |
| 0 | Linear regression (`lr`) |
| 1 | LSTM (`lstm`) |
| 2 | Chronos (`chronos`) |

Each task calls the standard SPCI runner, fitting one forest per sequence.
To run only the LSTM-base combination:

```bash
sbatch --array=1 sbatch/sbatch_run_spci/run_spci_air.sbatch
```

## Required inputs

Generate the base-predictor artifacts with the matching jobs under
[`../sbatch_run_base_predictor/`](../sbatch_run_base_predictor/) first.
Here, `P` is the selected predictor (`lr`, `lstm`, or `chronos`):

| Dataset | Required artifact, relative to the repository root |
| --- | --- |
| Air | `data/air-10_prediction/P/P_air-10_data.pkl` |
| Solar | `data/solar_prediction/P/P_nsdb-60m_data.pkl` |
| Sapflux | `data/sapflux-solo3-large/P/P_sapflux-solo3-large_data.pkl` |

Each task loads its dedicated
`configs/spci_configs/spci_{predictor}_{dataset}_config.yaml` for Air, Solar,
or Sapflux. Model and preprocessing settings come from that file; Solar no
longer inherits Air settings, and Sapflux predictors no longer share the
LSTM config. Artifact and output paths still follow the selected task.
The dispatcher sets `prediction_step: 1` to match the base-predictor jobs.

## Training and test split

`data.train_ratio` defines the complete training prefix of each saved
predictor's held-out sequence. The rest is the final test suffix. The
Air, Solar, and Sapflux configs use `train_ratio: 0.66` (about 66% training
and 34% test); the toy config uses `0.8`. Final evaluation fits one forest
on the whole training prefix and computes normalization statistics from
that prefix.

To migrate an older SPCI config, set the new `data.train_ratio` to the sum
of its old `train_ratio` and `valid_ratio`, then remove `data.valid_ratio`.
For example, `0.5 + 0.16` becomes `0.66`. This preserves approximately the
same final test boundary; integer rounding can shift it by one observation.
The runner rejects configs that still contain `data.valid_ratio`. Final
normalization now uses the complete training prefix, instead of the old
nominal training portion, so results can change after migration. SPCI has
no separate outer validation region. Configure tuning validation with
`tuning.model_selection_valid_ratio` as described below.

## Beta optimization

The experiment configs enable the SPCI tail-allocation search:

```yaml
model:
  optimize_beta: true
  beta_bins: 5
```

For each configured target pair, `1 - alpha` is its nominal coverage:
`max(pair) - min(pair)`. At each prediction time,
the fitted forest estimates residual quantiles for five equally spaced
`beta` values in `[0, alpha]`, including both endpoints. The chosen beta
minimizes the predicted residual interval width
`Q(1 - alpha + beta) - Q(beta)`. Adding those residual endpoints to the
base prediction gives the prediction interval. Selection uses predicted
widths and the available past residuals.

For example, `[0.95, 0.05]` requests 90% nominal coverage. With five bins,
the search compares `beta = [0, 0.025, 0.05, 0.075, 0.1]`. The configured
pair specifies coverage; the selected lower and upper quantile levels can
vary with prediction time. Increase `beta_bins` to use a finer search.

The forest remains fixed after fitting. Tuning and final evaluation use
the same beta-selection helper. Optimized intervals use standard Winkler
scoring with the nominal miscoverage `alpha` and penalties `2 / alpha`.
Selected beta values are saved with interval results for inspection.

Set `model.optimize_beta: false`, or omit that setting, to retain fixed
quantile endpoints and legacy scoring. The beta settings belong in the
base experiment YAML and are inherited by every hyperparameter trial.

## Outputs and overrides

Each task writes to `results/spci/{dataset}/{predictor}/`:

```text
resolved_config.yaml
log.pkl
summary_results.pkl
plots/
```

The default seed is `2026`. The dispatcher seeds Python, NumPy, and PyTorch
before calling the SPCI runner. Set `SPCI_SEED` or `SPCI_OUTPUT_ROOT` when
submitting; set `RUNPATH` if the repository is elsewhere. Repeating a
combination reuses its output directory, so use a separate output root to
retain another experiment:

```bash
sbatch --export=ALL,SPCI_SEED=17,SPCI_OUTPUT_ROOT=/path/to/results/spci_seed17 \
  sbatch/sbatch_run_spci/run_spci_air.sbatch
```

## Preview a task

From the repository root, inspect the resolved configuration with:

```bash
python -m sbatch.sbatch_run_spci.run_spci_job air --task-id 1 --dry-run
```

The preview requires `omegaconf` and does not write files, train, or require
Slurm or prediction artifacts.

## Hyperparameter tuning

Submit the tuning arrays after generating the predictor artifacts:

```bash
sbatch sbatch/sbatch_run_tuning/run_spci_air_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_spci_solar_tuning.sbatch
sbatch sbatch/sbatch_run_tuning/run_spci_sapflux_tuning.sbatch
```

Tasks 0/1/2 select LR/LSTM/Chronos. Each task evaluates 27 configurations on
three sequences for Air/Solar or five for Sapflux, using the dataset's
`spci_*_tuning_config.yaml` under
[`../../configs/spci_configs/`](../../configs/spci_configs/):

```yaml
grid:
  model.window_size: [100, 200, 500]
  model.n_estimators: [10, 20, 50]
  model.max_depth: [2, 5, 10]

tuning:
  num_sequences: 3
  delta_threshold: -0.01
  model_selection_valid_ratio: 0.15
```

### Fit and evaluation split

`tuning.model_selection_valid_ratio` reserves the **last fraction of the
training prefix** for hyperparameter evaluation. The forest fits on
the preceding training observations. For a sequence of length `N`, the
training prefix contains `floor(N * data.train_ratio)` observations; the
evaluation region is a tail of that prefix. Its fitting boundary is
`floor(nextafter(training_size * (1 - model_selection_valid_ratio), +inf))`.
`nextafter` stabilizes integer boundaries against floating-point rounding.
Normalization statistics come
only from the forest-fitting portion, and all window sizes evaluate the
same timestamps. Earlier observed residuals are available as lag features
during evaluation.

For example, with `data.train_ratio: 0.66` and
`tuning.model_selection_valid_ratio: 0.2`, the split is:

| Region | Fraction of the saved predictor's held-out sequence | Tuning use |
| --- | --- | --- |
| Forest fitting | First 52.8% | Fit the quantile forest |
| Model-selection validation | Next 13.2% | Evaluate hyperparameters |
| Final test | Last 34% | Unused |

Percentages are approximate because split boundaries use integer indices.
The supplied ratio of `0.15` instead fits on the first 56.1% and evaluates
on the next 9.9%. The final test suffix is excluded from tuning. Set
`model_selection_valid_ratio` in the tuning YAML strictly between zero and
one. Each sequence must have
enough fitting observations for the largest residual window in the grid.

### Selection and outputs

The tuner uses QR-CP's coverage filter and Winkler-score ranking: every
selected sequence and quantile pair must satisfy
`coverage - target_coverage > delta_threshold`. Eligible trials are ranked
by mean Winkler score, with equal sequence weights. The default jobs retain
the top three trials; if none passes coverage, `top_trials` is empty.

Each task writes to `results/tuning/spci_{base_predictor}_{dataset}/`:

```text
trial_0001/resolved_config.yaml
trial_0001/result.pkl
...
tuning_results.pkl
```

The summary contains `all_trials`, `top_trials`, the split protocol, and
`final_test_evaluated: false`. Trial records include per-sequence metrics
and seeds. Trials run sequentially, with each forest using at most the
allocated CPUs. Jobs activate `spci`, request four CPUs, 16 GB memory and
four hours, and support the `RUNPATH` repository override.

### Run or customize a search

The same runner can be called directly from the repository root:

```bash
python -m sbatch.sbatch_run_tuning.run_spci_tuning \
  --base-config configs/spci_configs/spci_lstm_air_config.yaml \
  --grid-config configs/spci_configs/spci_air_tuning_config.yaml \
  --save-dir 'results/tuning/spci_{base_predictor}_air' \
  --sequence-index 0 --top-k 3 --seed 2026
```

`{base_predictor}` is inferred from the base configuration or artifact
filename. `--sequence-index` starts a contiguous selection of sorted
sequence keys; `tuning.num_sequences: all` selects all sequences, and
`--sequence-key` selects one explicit key. The launchers forward command
arguments, so a separate tuning YAML can override the grid and ratio:

```bash
sbatch --array=1 sbatch/sbatch_run_tuning/run_spci_air_tuning.sbatch \
  --grid-config /path/to/my_spci_tuning_config.yaml \
  --save-dir /path/to/my_spci_search --top-k 5 --seed 17
```

Use a distinct save directory to retain another search. After selecting a
trial from `top_trials`, run its resolved configuration for final test
evaluation, substituting the selected trial number below:

```bash
python -m sbatch.sbatch_run_spci.run_spci \
  results/tuning/spci_lstm_air/trial_0001/resolved_config.yaml
```

The ordinary SPCI runner honors the saved seed, refits on the complete
training prefix, including the tuning validation tail, and evaluates the final
test region. Final outputs go to that trial's `final_run/` directory.
Change `saving_dir` in a copy of the selected YAML to choose another final
output directory. This tunes the current implementation, which fits one
forest per sequence and uses the base configuration's beta-optimization
settings during both tuning and final evaluation.

### Quantile-forest numerical stability

Beta optimization requests the 0th and 100th percentiles. In the exact
`sklearn_quantile==0.1.1` forest, accumulated float32 weights can leave the
100th percentile unset. The SPCI wrapper computes these two endpoints
directly from the minimum and maximum positive-weight training residuals
in the reached leaves. Interior quantile predictions are unchanged.

Above 10,000 fitting samples, both tuning and final evaluation use the
sampled quantile forest. In `sklearn_quantile==0.1.1`, float32 accumulation
can leave a sampled leaf value as NaN even when the training residuals are
finite. The shared SPCI model repairs affected leaves using their original
seeded draws and normalized float64 weights over that leaf's training
residuals. A repair emits a warning; each sequence's tuning result records
`num_repaired_leaves`. Leaves without valid training support still raise an
error, and validation predictions must remain finite.
