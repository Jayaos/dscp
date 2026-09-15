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

The RNN templates currently use an LSTM encoder. All tasks default to the
`partially_monotonic` prediction head, which evaluates requested quantiles
directly and prevents quantile crossing by construction. Select the legacy
cosine-embedding head, whose inference uses sampling and empirical
rearrangement, with `IQN_PREDICTION_HEAD`:

```bash
sbatch --export=ALL,IQN_PREDICTION_HEAD=cosine_embedding \
  sbatch/sbatch_run_iqn_cp/run_iqn_cp_air.sbatch
```

The same option applies to Solar and Sapflux. To run only the RNN/LSTM-base
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

The dispatcher uses the corresponding Air IQN-CP encoder template for Air
and Solar. Sapflux uses the existing Sapflux LSTM-base encoder template for
every base predictor, substituting the selected artifact path. These are
starting hyperparameters: they are not separate tuned configurations for
Solar or for the additional predictor/template combinations. All runs force
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
  --task-id 1 --prediction-head cosine_embedding --dry-run
```

This preview needs the Python configuration dependencies, but does not need
Slurm. Actual `.sbatch` array runs require the cluster environment described
above.
