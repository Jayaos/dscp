# IQN-CP Slurm jobs

Each dataset has one array job that runs both IQN-CP encoders with all three
base predictors. Submit from the repository root on the configured cluster:

```bash
sbatch sbatch/sbatch_run_iqn_cp/run_iqn_cp_air.sbatch
sbatch sbatch/sbatch_run_iqn_cp/run_iqn_cp_solar.sbatch
sbatch sbatch/sbatch_run_iqn_cp/run_iqn_cp_sapflux.sbatch
```

The scripts require Slurm, the configured account/partition and repository
path, and the cluster's `anaconda3` module with the `dscp` conda environment.
They request a four-hour wall-time by default. Adjust it for another cluster
or experiment, for example with `sbatch --time=08:00:00 ...`.

## Array tasks and prediction-head choice

Each submission creates tasks `0-5` with this mapping:

| Task ID | IQN-CP encoder | Base predictor |
| --- | --- | --- |
| 0 | RNN | Linear regression (`lr`) |
| 1 | RNN | LSTM (`lstm`) |
| 2 | RNN | Chronos (`chronos`) |
| 3 | Transformer | Linear regression (`lr`) |
| 4 | Transformer | LSTM (`lstm`) |
| 5 | Transformer | Chronos (`chronos`) |

The RNN templates currently use an LSTM encoder. Each task uses
`model.prediction_head` from its selected YAML config by default:

```yaml
model:
  prediction_head: cosine_embedding # or partially_monotonic
```

The partially monotonic head prevents quantile crossing by construction.
For the cosine head, YAML `model.interval_mode` selects direct evaluation or
sampling and empirical rearrangement. Configs without a head selector retain
the core runner's legacy `cosine_embedding` default.

To explicitly override the YAML choice for a submission, set
`IQN_PREDICTION_HEAD`:

```bash
sbatch --export=ALL,IQN_PREDICTION_HEAD=cosine_embedding \
  sbatch/sbatch_run_iqn_cp/run_iqn_cp_air.sbatch
```

The same option applies to Solar and Sapflux. Leave `IQN_PREDICTION_HEAD`
unset or empty to use the YAML choice; no environment override is needed for
cosine mode. Direct dispatcher calls accept the optional `--prediction-head`
override as well. To run only the RNN/LSTM-base
combination, restrict the array:

```bash
sbatch --array=1 sbatch/sbatch_run_iqn_cp/run_iqn_cp_air.sbatch
```

## Required base-predictor artifacts

Generate the saved base-predictor artifacts before running IQN-CP. The
matching jobs under [`../sbatch_run_base_predictor/`](../sbatch_run_base_predictor/)
write the paths below. If invoking their Python launchers directly, supply an
explicit `--save-dir` to match these paths. Here, `P` is `lr`, `lstm`, or
`chronos`, and paths are relative to the repository root:

| Dataset argument | Required artifact |
| --- | --- |
| `air` | `data/air-10_prediction/P/P_air-10_data.pkl` |
| `solar` | `data/solar_prediction/P/P_nsdb-60m_data.pkl` |
| `sapflux` | `data/sapflux-solo3-large/P/P_sapflux-solo3-large_data.pkl` |

Each task needs only its selected predictor's artifact; a complete dataset
array needs all three. IQN-CP consumes the saved held-out forecasts and
observations.

## Configuration and outputs

Air, Solar, and Sapflux jobs load
`configs/iqn_cp_configs/iqn_{encoder}_{predictor}_{dataset}_config.yaml`, where
`encoder` is `rnn` or `transformer`, `predictor` is `lr`, `lstm`, or `chronos`,
and `dataset` is `air`, `solar`, or `sapflux`. Each task preserves its selected
config's model and training settings instead of loading a generic template.

`training.tau_mode` comes from that YAML: `sampled_quantiles` (the default)
trains across uniformly sampled levels, while `target_quantiles` trains on all
distinct endpoints in `model.target_quantiles` for each observation. The latter
ignores `model.num_taus` for training, but
`training.validation_loss: sampled_quantiles` still uses it. Validation and
interval settings remain independent. For a cosine
endpoint-learning diagnostic, use:

```yaml
model:
  prediction_head: cosine_embedding
  target_quantiles:
    - [0.95, 0.05]
  interval_mode: direct
training:
  tau_mode: target_quantiles
  validation_loss: target_quantiles
```

No prediction-head environment override is needed for this YAML configuration.
Target-only training does not supervise other quantile
levels, so combining it with cosine sampling intervals emits a warning without
changing the chosen modes. Use a separate output directory when comparing
training modes; `tau_mode` does not change the checkpoint architecture or add
an automatic output-path component.

The six Solar configs initially match their corresponding Air hyperparameters;
only their prediction artifact and output paths differ. They are separate files
so future Solar-specific settings can be made without changing Air. The LR and
Chronos Sapflux configs initially copy the corresponding LSTM Sapflux encoder
settings, with their own predictor data and output paths. Each is loaded
independently; no generic or shared template is required. All runs force
`prediction_step: 1`.

Each run writes its resolved settings and outputs under:

```text
results/iqn_cp/{dataset}/{predictor}/{encoder}/{prediction_head}/
    resolved_config.yaml
    *_model.pt
    log.pkl
    summary_results.pkl
    plots/
```

The encoder directory is `rnn` or `transformer`; the prediction-head
directory is `partially_monotonic` or `cosine_embedding`. Keeping the two
heads separate is required because their checkpoint layouts differ.
Repeating the same combination reuses its output directory. Do not run the
same combination concurrently, because both jobs would write the same files.
For independent submissions or seed sweeps, set a unique `IQN_OUTPUT_ROOT`,
for example `--export=ALL,IQN_OUTPUT_ROOT="$PWD/results/iqn_cp/seed-7"`.

The dispatcher defaults to seed `2026`, configurable with `--seed`. For array
submissions, the corresponding environment variables are `IQN_OUTPUT_ROOT`
and `IQN_SEED`; `RUNPATH` overrides the repository path. The dispatcher seeds
Python, NumPy, and PyTorch before model construction. GPU execution is not
promised to be bitwise deterministic across platforms.

## Preview a task

From the repository root in a Bash shell, inspect the resolved configuration
without submitting a job, writing results, training, importing PyTorch, or
requiring the data artifact:

```bash
PYTHONPATH="$PWD:$PWD/sbatch:${PYTHONPATH:-}" \
  python -m sbatch_run_iqn_cp.run_iqn_cp_job air \
  --task-id 1 --dry-run
```

This preview uses the YAML head; add `--prediction-head cosine_embedding` or
`--prediction-head partially_monotonic` only to override it.
It needs the Python configuration dependencies, but does not need
Slurm. Actual `.sbatch` array runs require the cluster environment described
above.
