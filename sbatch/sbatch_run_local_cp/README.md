# Local-CP Slurm jobs

Each dataset has one array job that runs both Local-CP encoders with all three
base predictors. Submit from the repository root on the configured cluster:

```bash
sbatch sbatch/sbatch_run_local_cp/run_local_cp_air.sbatch
sbatch sbatch/sbatch_run_local_cp/run_local_cp_solar.sbatch
sbatch sbatch/sbatch_run_local_cp/run_local_cp_sapflux.sbatch
```

The scripts request one V100 GPU and four CPUs, using the configured Slurm
account and `inferno` QoS. Air uses 16 GB memory and a two-hour limit; Sapflux
uses 32 GB and two hours; Solar uses 32 GB and four hours.
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

The dispatcher selects the matching dataset, encoder, and predictor
configuration, where `{encoder}` is `rnn` or `transformer` and `{predictor}`
is `lr`, `lstm`, or `chronos`:

| Dataset | Template |
| --- | --- |
| Air | `configs/lcp_configs/lcp_{encoder}_{predictor}_air_config.yaml` |
| Solar | `configs/lcp_configs/lcp_{encoder}_{predictor}_solar_config.yaml` |
| Sapflux | `configs/lcp_configs/lcp_{encoder}_{predictor}_sapflux_config.yaml` |

The Solar and Sapflux configurations were initialized from the corresponding
Air configuration, preserving all hyperparameters and data split settings.
Only the dataset path and ordinary output directory differ. These are separate
YAML files; later edits to an Air configuration do not automatically update
the other datasets. The arrays run the selected experiment configurations.
The dispatcher sets the selected data path, `model.prediction_step: 1`,
`saving_dir` and seed, and defaults `device` to `0` if absent.

Edit the corresponding YAML templates to configure Local-CP. The dispatcher
preserves these settings:

- `model.training_quantiles`: levels used for encoder training and validation
  checkpoint selection with pinball loss.
- `model.target_quantiles`: interval quantile pairs used for inference and
  evaluation, independently of the training levels.
- `model.similarity_fn` and `model.temperature`: similarity weighting. Current
  templates use `cos_similarity` and `10.0`. Supported choices are `dot_product`,
  `cos_similarity` and `euclidean`. Dot product and cosine multiply similarity
  by temperature before softmax; Euclidean uses negative squared distance
  divided by a positive temperature before softmax.
- `model.rolling_calibration`: `true` updates the calibration pool after each
  observed prediction; `false` keeps the initial calibration pool fixed.
  Current templates use `true`.

## Epanechnikov experiment

The existing arrays retain softmax weighting. Separate arrays opt into the
Epanechnikov kernel on cosine-derived chord distance, using the matching
non-tuning configuration from `configs/lcp_epanechnikov_kernel_configs/`:

```bash
sbatch sbatch/sbatch_run_local_cp/run_local_cp_epanechnikov_air.sbatch
sbatch sbatch/sbatch_run_local_cp/run_local_cp_epanechnikov_solar.sbatch
sbatch sbatch/sbatch_run_local_cp/run_local_cp_epanechnikov_sapflux.sbatch
```

These arrays use the same task mapping and resources as their corresponding
softmax arrays. Each task loads
`lcp_{encoder}_epanechnikov_{predictor}_{dataset}_config.yaml`, including its
training settings, rolling calibration, and selected `model.bandwidth`.
The dispatcher sets `model.kernel: epanechnikov` and
`model.similarity_fn: cos_similarity`. Temperature is unused for this kernel.
`LOCAL_CP_BANDWIDTH` is optional; when set, it overrides the YAML bandwidth
for every task in that submission.
The existing `model.sampling_num` and sampled residual quantiles are retained.
An empty kernel neighborhood returns zero residual quantiles, matching the
manuscript's point-mass-at-zero fallback. This kernel satisfies the kernel
conditions in A.22; it does not establish the local-mass or other asymptotic
assumptions.

Preview a task using its saved bandwidth:

```bash
PYTHONPATH="$PWD:$PWD/sbatch:${PYTHONPATH:-}" \
  python -m sbatch_run_local_cp.run_local_cp_job air \
  --task-id 1 --kernel epanechnikov --dry-run
```

`--kernel epanechnikov` selects the Epanechnikov templates; otherwise the
dispatcher uses `configs/lcp_configs/`. `--kernel` also overrides `model.kernel`
in the selected template; absent both, the default is `softmax`.
`--bandwidth` can override the template's `model.bandwidth`; the resolved value
must be finite and positive. The default Epanechnikov output root is
`results/local_cp_epanechnikov/bandwidth_<value>/seed_<seed>/`, with the
template's dataset/predictor/encoder/run suffix preserved. `LOCAL_CP_OUTPUT_ROOT`
or `--output-root` can override this root. Each job trains its own encoder;
these launchers do not reuse a frozen checkpoint across submitted jobs.

## Epanechnikov on CPU

The CPU arrays run independent sequences concurrently, with one CPU thread
per worker:

```bash
sbatch sbatch/sbatch_run_local_cp/run_local_cp_epanechnikov_air_cpu.sbatch
sbatch sbatch/sbatch_run_local_cp/run_local_cp_epanechnikov_solar_cpu.sbatch
sbatch sbatch/sbatch_run_local_cp/run_local_cp_epanechnikov_sapflux_cpu.sbatch
```

| Dataset | CPUs / sequence workers | Memory | Time limit |
| --- | --- | --- | --- |
| Air | 12 | 48 GB | 30 minutes |
| Solar | 10 | 40 GB | 60 minutes |
| Sapflux | 12 | 48 GB | 30 minutes |

These allocations apply to **each array task**. Each dataset still has six
encoder/predictor tasks (`0-5`). A worker processes one sequence at a time and
takes another when it finishes; if there are fewer sequences than allocated
CPUs, only the needed workers run. Timesteps within a sequence remain ordered
for rolling calibration. Memory is four GB per allocated CPU.

The scripts request no GPU and pass `--cpu-workers "$SLURM_CPUS_PER_TASK"` to
the dispatcher. This forces `device: cpu` in the resolved run configuration
and uses a separate process per sequence worker. The shared YAML templates
retain their GPU settings. PyTorch and numerical-library threads are limited
to one per worker to avoid oversubscribing the allocation.

Each sequence uses the run seed plus its position in the prediction artifact
(modulo `2**32`), recorded as `sequence_seed` in its log. This makes CPU results
reproducible across worker counts. Workers write separate model checkpoints;
the parent merges sequence logs in artifact order and writes the combined
summary and plots once all sequences finish.
The GPU runner uses one advancing random-number stream, so matching base seeds
do not imply identical fitted models between CPU and GPU runs.

The CPU arrays use the same Epanechnikov templates and optional
`LOCAL_CP_BANDWIDTH`, `LOCAL_CP_SEED`, `LOCAL_CP_OUTPUT_ROOT`, and `RUNPATH`
overrides. Their default result root is
`results/local_cp_epanechnikov_cpu/bandwidth_<value>/seed_<seed>/`, followed by
the template's dataset/predictor/encoder/run suffix. The existing GPU scripts
retain their behavior.

Preview the resolved CPU configuration without starting workers:

```bash
PYTHONPATH="$PWD:$PWD/sbatch:${PYTHONPATH:-}" \
  python -m sbatch_run_local_cp.run_local_cp_job air \
  --task-id 1 --kernel epanechnikov --cpu-workers 12 --dry-run
```

## Outputs and reproducibility

Each run writes its resolved settings and outputs under:

```text
results/local_cp/{dataset}/{predictor}/{encoder}/{configured_run_suffix}/
    resolved_config.yaml
    *_model.pt
    log.pkl
    summary_results.pkl
    plots/
```

The encoder directory is `rnn` or `transformer`. The dispatcher preserves the
template's run suffix, including a timestamp when configured. A template without
a unique run suffix reuses its output directory; use a separate `--output-root`
to retain those runs. The dispatcher defaults to seed `2026`, configurable with
`--seed`.

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
