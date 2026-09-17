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

Air and Solar use the corresponding predictor's Air SPCI configuration
template. Sapflux uses `spci_lstm_sapflux_config.yaml` for all three
predictors, substituting the selected artifact path. These are starting
hyperparameters, with no additional tuning for Solar or Sapflux combinations.
The dispatcher sets `prediction_step: 1` to match the base-predictor jobs.

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
three sequences, using the dataset's `spci_*_tuning_config.yaml` under
[`../../configs/spci_configs/`](../../configs/spci_configs/):

```yaml
grid:
  model.window_size: [100, 200, 300]
  model.n_estimators: [10, 50, 100]
  model.max_depth: [2, 5, 10]

tuning:
  num_sequences: 3
  delta_threshold: -0.01
  model_selection_valid_ratio: 0.15
```

### Fit and evaluation split

`tuning.model_selection_valid_ratio` reserves the **last fraction of the
nominal training set** for hyperparameter evaluation. The forest fits on
the preceding training observations. For a sequence of length `N`, the
nominal training prefix ends at `floor(N * data.train_ratio)`; the
evaluation region is a tail of that prefix. Normalization statistics come
only from the forest-fitting portion, and all window sizes evaluate the
same timestamps. Earlier observed residuals are available as lag features
during evaluation.

For example, with `data.train_ratio: 0.5` and
`tuning.model_selection_valid_ratio: 0.2`, the split is:

| Region | Fraction of the saved predictor's held-out sequence | Tuning use |
| --- | --- | --- |
| Forest fitting | First 40% | Fit the quantile forest |
| Model-selection validation | Next 10% | Evaluate hyperparameters |
| Nominal outer validation | Next 16% (`data.valid_ratio: 0.16`) | Unused |
| Final test | Last 34% | Unused |

Percentages are approximate because split boundaries use integer indices.
The supplied ratio of `0.15` instead fits on the first 42.5% and evaluates
on the next 7.5%. `data.valid_ratio` defines the outer validation region;
it does not determine the hyperparameter evaluation size. Set the ratio in
the tuning YAML strictly between zero and one. Each sequence must have
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
nominal training plus outer validation regions, and evaluates the final
test region. Final outputs go to that trial's `final_run/` directory.
Change `saving_dir` in a copy of the selected YAML to choose another final
output directory. This tunes the current implementation, which fits one
forest per sequence and uses the configured fixed quantile levels.
