# NexCP Slurm jobs

Submit from the repository root:

```bash
sbatch sbatch/sbatch_run_nexcp/run_nexcp_air.sbatch
sbatch sbatch/sbatch_run_nexcp/run_nexcp_solar.sbatch
sbatch sbatch/sbatch_run_nexcp/run_nexcp_sapflux.sbatch
```

Each dataset job is an array: task **0 = LR**, **1 = LSTM**, **2 = Chronos**.
For example, submit only the LSTM Air run with:

```bash
sbatch --array=1 sbatch/sbatch_run_nexcp/run_nexcp_air.sbatch
```

The scripts use the existing cluster account, Inferno QoS, and notification
address. Each task requests one CPU, 4 GB RAM, and four hours; NexCP evaluates
saved forecasts sequentially and needs no GPU. Override resources using
`sbatch --mem=SIZE --time=HH:MM:SS` as needed.

The jobs load `anaconda3` and activate `dscp` (see
[`../../envs/env-dscp.yml`](../../envs/env-dscp.yml)).
Set `NEXCP_ENV` or `NEXCP_CONDA_MODULE` to select another installed
environment or module. The repository location defaults to the submission
directory; set `DSCP_RUNPATH` (or `RUNPATH`) when submitting elsewhere:

```bash
sbatch --export=ALL,DSCP_RUNPATH=/path/to/dscp,NEXCP_ENV=dscp \
  sbatch/sbatch_run_nexcp/run_nexcp_air.sbatch
```

## Inputs and configurations

Generate the saved forecasts first using the corresponding jobs under
[`../sbatch_run_base_predictor/`](../sbatch_run_base_predictor/).
Here `P` is `lr`, `lstm`, or `chronos`:

| Dataset | Required artifact, relative to the repository root |
| --- | --- |
| Air | `data/air-10_prediction/P/P_air-10_data.pkl` |
| Solar | `data/solar_prediction/P/P_nsdb-60m_data.pkl` |
| Sapflux | `data/sapflux-solo3-large/P/P_sapflux-solo3-large_data.pkl` |

Each task invokes the existing runner with
`configs/nexcp_configs/nexcp_P_DATASET_config.yaml`.
The presets use 66% initial calibration, `rho: 0.99`, a maximum of 200
past residuals, and a 90% interval. The added Solar and Sapflux presets share
these starting settings; they have not been tuned for those datasets.

## Outputs

The configuration's `saving_dir` controls the output location:

| Dataset | Output directory |
| --- | --- |
| Air | `results/nexcp_P_air/` |
| Solar | `results/nexcp_P_solar/` |
| Sapflux | `results/nexcp_P_sapflux_solo3_large/` |

Runs save `log.pkl`, `summary_results.pkl`, and `plots/`.
Each array task has its own result directory and Slurm log
(`Report-nexcp-DATASET-JOBID_TASKID.out`).
Repeated runs reuse the configured result directory.

To run a configuration directly without Slurm:

```bash
python -m sbatch.sbatch_run_nexcp.run_nexcp configs/nexcp_configs/nexcp_lr_air_config.yaml
```
