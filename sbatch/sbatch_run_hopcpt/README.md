# HopCPT Slurm jobs

Submit one array per dataset from the repository root on the configured cluster:

```bash
sbatch sbatch/sbatch_run_hopcpt/run_hopcpt_air.sbatch
sbatch sbatch/sbatch_run_hopcpt/run_hopcpt_solar.sbatch
sbatch sbatch/sbatch_run_hopcpt/run_hopcpt_sapflux.sbatch
```

The scripts follow the cluster settings in `../sbatch_run_qr_cp/`: the same
account, Inferno QoS, V100 GPU, four CPUs, 16 GB host memory, repository path,
and notification address. They activate the `hopcpt` conda environment defined
in [`../../envs/env-hopcpt.yml`](../../envs/env-hopcpt.yml), which includes
`hopfield-layers`. Each task requests four hours for the templates' 3,000
training epochs per sequence. Adjust the limit with `sbatch --time=HH:MM:SS`
as needed; runtime depends on sequence count and length.

## Array tasks

| Task ID | Base predictor |
| --- | --- |
| 0 | Linear regression (`lr`) |
| 1 | LSTM (`lstm`) |
| 2 | Chronos (`chronos`) |

Each task calls the standard HopCPT runner, training one model per sequence
on its allocated GPU. To run only the LSTM-base combination:

```bash
sbatch --array=1 sbatch/sbatch_run_hopcpt/run_hopcpt_air.sbatch
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

Each dataset and predictor has an explicit configuration under
`configs/hopcpt_configs/hopcpt_{predictor}_{dataset}_config.yaml`.
Solar and Sapflux use the original publication's Ridge/LR HopCPT settings
for all three base predictors: learning rate `0.01`, 3,000 epochs,
validation every 5 epochs, no dropout, and no temporal encoding.
Solar uses a memory size of 5,000; Sapflux uses 4,000. These settings
come from the authors' Ridge experiment commands and their configuration
defaults, without additional tuning for LSTM or Chronos.

The standard runner continues to train one model per sequence. In particular,
this does not reproduce the original Solar training batch of four sequences.
The dispatcher sets `prediction_step: 1`, `device: 0`, and disables internal
multi-GPU processing to match each task's single GPU allocation.

## Outputs and overrides

Each task writes to `results/hopcpt/{dataset}/{predictor}/`:

```text
resolved_config.yaml
*_model.pt
log.pkl
summary_results.pkl
plots/
```

The default seed is `2026`. Set `HOPCPT_SEED` or `HOPCPT_OUTPUT_ROOT` when
submitting; set `RUNPATH` if the repository is elsewhere. Repeating a
combination reuses its output directory, so use a separate output root to
retain another experiment:

```bash
sbatch --export=ALL,HOPCPT_SEED=17,HOPCPT_OUTPUT_ROOT=/path/to/results/hopcpt_seed17 \
  sbatch/sbatch_run_hopcpt/run_hopcpt_air.sbatch
```

## Preview a task

From the repository root, inspect the resolved configuration with:

```bash
python -m sbatch.sbatch_run_hopcpt.run_hopcpt_job air --task-id 1 --dry-run
```

The preview requires `omegaconf` and does not write files, train, or require
Slurm, CUDA, or prediction artifacts.
