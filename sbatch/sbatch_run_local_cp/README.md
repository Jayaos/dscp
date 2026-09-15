# Local-CP Slurm jobs

Each dataset has one array job that runs both Local-CP encoders with all three
base predictors. Submit from the repository root on the configured cluster:

```bash
sbatch sbatch/sbatch_run_local_cp/run_local_cp_air.sbatch
sbatch sbatch/sbatch_run_local_cp/run_local_cp_solar.sbatch
sbatch sbatch/sbatch_run_local_cp/run_local_cp_sapflux.sbatch
```

The scripts follow the QR-CP jobs: one V100 GPU, four CPUs, 16 GB memory and a
10-minute time limit, using the configured Slurm account and `inferno` QoS.
They require the cluster's `anaconda3` module and `dscp` conda environment.
Adjust the account, resource settings, repository path and notification
address for another cluster before submitting.

Override the inherited time limit for longer runs:

```bash
sbatch --time=04:00:00 sbatch/sbatch_run_local_cp/run_local_cp_air.sbatch
```

## Array tasks

Each submission creates tasks `0-5` with this mapping:

| Task ID | Local-CP encoder | Base predictor |
| --- | --- | --- |
| 0 | RNN | Linear regression (`lr`) |
| 1 | RNN | LSTM (`lstm`) |
| 2 | RNN | Chronos (`chronos`) |
| 3 | Transformer | Linear regression (`lr`) |
| 4 | Transformer | LSTM (`lstm`) |
| 5 | Transformer | Chronos (`chronos`) |

The RNN templates currently use an LSTM encoder. To run only the RNN/LSTM-base
combination, restrict the array:

```bash
sbatch --array=1 sbatch/sbatch_run_local_cp/run_local_cp_air.sbatch
```

## Required base-predictor artifacts

Generate the saved base-predictor artifacts before running Local-CP. The
matching jobs under [`../sbatch_run_base_predictor/`](../sbatch_run_base_predictor/)
write the paths below. If invoking their Python launchers directly, supply an
explicit `--save-dir` to match these paths. Here, `P` is `lr`, `lstm`, or
`chronos`, and paths are relative to the repository root:

| Dataset argument | Required artifact |
| --- | --- |
| `air` | `data/air-10_prediction/P/P_air-10_data.pkl` |
| `solar` | `data/solar_prediction/P/P_nsdb-60m_data.pkl` |
| `sapflux` | `data/sapflux-solo3-large/P/P_sapflux-solo3-large_data.pkl` |

Each task needs its selected predictor's artifact; a complete dataset array
needs all three. Local-CP consumes the saved held-out forecasts and observations.

## Configuration

The dispatcher selects these templates for every base predictor, where
`{encoder}` is `rnn` or `transformer`:

| Dataset | Template |
| --- | --- |
| Air and Solar | `configs/lcp_configs/lcp_{encoder}_chronos_air_config.yaml` |
| Sapflux | `configs/lcp_configs/lcp_{encoder}_lstm_sapflux_config.yaml` |

These supply starting hyperparameters; the arrays do not perform a
hyperparameter search for the additional dataset/predictor combinations.
The dispatcher sets the selected data path, `model.prediction_step: 1`,
`saving_dir` and seed, and defaults `device` to `0` if absent.

Edit the corresponding YAML templates to configure Local-CP. The dispatcher
preserves these settings:

- `model.training_quantiles`: levels used for encoder training and validation
  checkpoint selection with pinball loss.
- `model.target_quantiles`: interval quantile pairs used for inference and
  evaluation, independently of the training levels.
- `model.similarity_fn` and `model.temperature`: similarity weighting. Current
  templates use `dot_product` and `0.1`. Supported choices are `dot_product`,
  `cos_similarity` and `euclidean`. Dot product and cosine multiply similarity
  by temperature before softmax; Euclidean uses negative squared distance
  divided by a positive temperature before softmax.
- `model.rolling_calibration`: `true` updates the calibration pool after each
  observed prediction; `false` keeps the initial calibration pool fixed.
  Current templates use `true`.

## Outputs and reproducibility

Each run writes its resolved settings and outputs under:

```text
results/local_cp/{dataset}/{predictor}/{encoder}/
    resolved_config.yaml
    *_model.pt
    log.pkl
    summary_results.pkl
    plots/
```

The encoder directory is `rnn` or `transformer`. Repeating the same combination
reuses its output directory. Use a separate `--output-root` for each experiment
or seed whose results should be retained. The dispatcher defaults to seed
`2026`, configurable with `--seed`.

For array submissions, use `LOCAL_CP_OUTPUT_ROOT` and `LOCAL_CP_SEED`;
`RUNPATH` overrides the repository path. For example:

```bash
sbatch --export=ALL,LOCAL_CP_OUTPUT_ROOT=results/local_cp_seed2027,LOCAL_CP_SEED=2027 \
  sbatch/sbatch_run_local_cp/run_local_cp_air.sbatch
```

## Preview a task

From the repository root in a Bash shell, inspect the resolved configuration
without submitting a job, writing results, training, or requiring the data
artifact:

```bash
PYTHONPATH="$PWD:$PWD/sbatch:${PYTHONPATH:-}" \
  python -m sbatch_run_local_cp.run_local_cp_job air \
  --task-id 1 --dry-run
```

This preview requires OmegaConf, but does not need Slurm. Actual `.sbatch`
array runs require the cluster environment described above.
