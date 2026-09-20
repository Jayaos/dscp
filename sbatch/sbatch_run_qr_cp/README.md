# QR-CP Slurm jobs

Each dataset has one array job that runs both QR-CP encoders with all three
base predictors. Submit from the repository root on the configured cluster:

```bash
sbatch sbatch/sbatch_run_qr_cp/run_qr_cp_air.sbatch
sbatch sbatch/sbatch_run_qr_cp/run_qr_cp_solar.sbatch
sbatch sbatch/sbatch_run_qr_cp/run_qr_cp_sapflux.sbatch
```

The scripts require Slurm, the configured account/partition and repository
path, and the cluster's `anaconda3` module with the `dscp` conda environment.
Adjust those settings for another cluster before submitting.

## Array tasks and head choice

Each submission creates tasks `0-5` with this mapping:

| Task ID | QR-CP encoder | Base predictor |
| --- | --- | --- |
| 0 | RNN | Linear regression (`lr`) |
| 1 | RNN | LSTM (`lstm`) |
| 2 | RNN | Chronos (`chronos`) |
| 3 | Transformer | Linear regression (`lr`) |
| 4 | Transformer | LSTM (`lstm`) |
| 5 | Transformer | Chronos (`chronos`) |

The RNN templates currently use an LSTM encoder. All tasks default to the
existing `nondecreasing` quantile heads. Select direct independent heads,
which allow quantile crossing, with `QR_HEAD_TYPE`:

```bash
sbatch --export=ALL,QR_HEAD_TYPE=independent sbatch/sbatch_run_qr_cp/run_qr_cp_air.sbatch
```

The same option applies to Solar and Sapflux. To run only the RNN/LSTM-base
combination, restrict the array:

```bash
sbatch --array=1 sbatch/sbatch_run_qr_cp/run_qr_cp_air.sbatch
```

## Required base-predictor artifacts

Generate the saved base-predictor artifacts before running QR-CP. The matching
jobs under [`../sbatch_run_base_predictor/`](../sbatch_run_base_predictor/)
write the paths below. If invoking their Python launchers directly, supply an
explicit `--save-dir` to match these paths. Here, `P` is `lr`, `lstm`, or
`chronos`, and paths are relative to the repository root:

| Dataset argument | Required artifact |
| --- | --- |
| `air` | `data/air-10_prediction/P/P_air-10_data.pkl` |
| `solar` | `data/solar_prediction/P/P_nsdb-60m_data.pkl` |
| `sapflux` | `data/sapflux-solo3-large/P/P_sapflux-solo3-large_data.pkl` |

Each task needs its selected predictor's artifact; a complete dataset array
needs all three. QR-CP consumes the saved held-out forecasts and observations.

## Configuration and outputs

Each task loads its dedicated
`configs/qr_cp_configs/qr_{encoder}_{predictor}_{dataset}_config.yaml` for Air,
Solar, or Sapflux. Model, training, and preprocessing settings come from that
file; Solar no longer inherits Air settings, and Sapflux predictors no longer
share the LSTM config. The dispatcher resolves the matching forecast artifact
and output directory. All runs use `prediction_step: 1`.

Each run writes its resolved settings and outputs under:

```text
results/qr_cp/{dataset}/{predictor}/{encoder}/{head_type}/
    resolved_config.yaml
    *_model.pt
    log.pkl
    summary_results.pkl
    plots/
```

The encoder directory is `rnn` or `transformer`. Separate head directories
retain both methods' results. Repeating the same combination reuses its output
directory; use the dispatcher's `--output-root` for a separate experiment.
The dispatcher defaults to seed `2026`, configurable with `--seed`.
For array submissions, the corresponding environment variables are
`QR_OUTPUT_ROOT` and `QR_SEED`; `RUNPATH` overrides the repository path.

## Preview a task

From the repository root in a Bash shell, inspect the resolved configuration
without submitting a job, writing results, training, or requiring the data
artifact:

```bash
PYTHONPATH="$PWD:$PWD/sbatch:${PYTHONPATH:-}" \
  python -m sbatch_run_qr_cp.run_qr_cp_job air \
  --task-id 1 --head-type independent --dry-run
```

This preview needs the Python configuration dependencies, but does not need
Slurm. Actual `.sbatch` array runs require the cluster environment described
above.
