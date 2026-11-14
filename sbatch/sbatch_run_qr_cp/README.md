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

The RNN templates currently use an LSTM encoder. Each task uses `model.head_type`
from its YAML template; omitting that setting falls back to `nondecreasing`.
An explicit `QR_HEAD_TYPE` overrides the YAML for Slurm submissions. For example,
select direct independent heads, which allow quantile crossing, with:

```bash
sbatch --export=ALL,QR_HEAD_TYPE=independent sbatch/sbatch_run_qr_cp/run_qr_cp_air.sbatch
```

The Python dispatcher likewise only overrides the YAML when `--head-type` is
supplied. Its startup message and `resolved_config.yaml` show the selected head.
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

## CPU comparison using CT-SSF's saved LSTM forecasts

These separate jobs use the same `lstm_model_saved` forecast files as CT-SSF
and the saved-LSTM Local-CP jobs:

```bash
sbatch sbatch/sbatch_run_qr_cp/run_qr_cp_lstm_saved_air_cpu.sbatch
sbatch sbatch/sbatch_run_qr_cp/run_qr_cp_lstm_saved_solar_cpu.sbatch
sbatch sbatch/sbatch_run_qr_cp/run_qr_cp_lstm_saved_sapflux_cpu.sbatch
```

They default to **task 1: RNN encoder with LSTM base predictions**, matching the
current saved-LSTM Local-CP scripts. Select task 4 for Transformer or both tasks:

```bash
sbatch --array=4 sbatch/sbatch_run_qr_cp/run_qr_cp_lstm_saved_air_cpu.sbatch
sbatch --array=1,4 sbatch/sbatch_run_qr_cp/run_qr_cp_lstm_saved_air_cpu.sbatch
```

Other task IDs are rejected. Allocations apply per array task and match the
current saved-LSTM Local-CP CPU jobs:

| Dataset | Forecast artifact | CPU cores | Memory | Time limit |
| --- | --- | --- | --- | --- |
| Air | `data/air-10_prediction/lstm_model_saved/lstm_air-10_data.pkl` | 12 | 48 GB | 4 hours |
| Solar | `data/solar_prediction/lstm_model_saved/lstm_nsdb-60m_data.pkl` | 10 | 40 GB | 6 hours |
| Sapflux | `data/sapflux-solo3-large/lstm_model_saved/lstm_sapflux-solo3-large_data.pkl` | 12 | 48 GB | 4 hours |

The jobs use `dscp`, request no GPU, and pass `--device cpu` and
`--num-threads "$SLURM_CPUS_PER_TASK"`. The existing QR-CP runner processes series
sequentially, using these cores for PyTorch tensor operations. This differs from
Local-CP's separate single-threaded sequence workers; it retains QR-CP's existing
training loop and seed behavior.

Each task still loads its matching
`configs/qr_cp_configs/qr_{encoder}_lstm_{dataset}_config.yaml`. QR-CP model,
training, preprocessing, split, plotting, and quantile-head settings are
preserved. `QR_HEAD_TYPE` can explicitly override the template as before.
The original GPU jobs and YAML templates are unchanged.

QR-CP reads the saved `.pkl` observations and predictions, without loading or
retraining the LSTM base model. It does not need the `.pt` checkpoint or a
completed CT-SSF run. Once the base-predictor artifacts exist, the comparison
methods can run independently.

The new default output root is separate from ordinary QR-CP runs:

```text
results/qr_cp_lstm_saved_cpu/<dataset>/lstm/<encoder>/<head_type>/<configured_run_suffix>/
```

`QR_OUTPUT_ROOT`, `QR_SEED`, `QR_HEAD_TYPE`, and `RUNPATH` retain their existing
override behavior. The dispatcher accepts `--data-path` to override the forecast
file (relative to the repository root when not absolute), and `--device` and
`--num-threads` to override runtime settings. Preview without training:

```bash
PYTHONPATH="$PWD:$PWD/sbatch:${PYTHONPATH:-}" \
  python -m sbatch_run_qr_cp.run_qr_cp_job air \
  --task-id 1 --device cpu --num-threads 12 \
  --data-path data/air-10_prediction/lstm_model_saved/lstm_air-10_data.pkl \
  --output-root results/qr_cp_lstm_saved_cpu --dry-run
```

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
