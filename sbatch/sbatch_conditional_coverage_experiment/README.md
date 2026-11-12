# Conditional coverage Slurm workflow

Run these commands from the repository root on the cluster. Both scripts load
the `anaconda3` module, activate the `dscp` environment, and change into
`RUNPATH`. By default, this is the Slurm submission directory. Export `RUNPATH`
with your checkout's absolute path if submitting from elsewhere. The Slurm
account, QoS, and resource directives follow the
repository's existing jobs; adjust them for your allocation when needed.

## Configure the experiment

Choose a grouping preset, such as
`configs/conditional_coverage_configs/air_lstm.yaml`. Check its source
`data_path`, `output_dir`, `clustering.k`, and comparison split settings.
The grouping `output_dir` is a base directory. Step 1 saves the complete
artifact in `<output_dir>/k<K>`, using the configured `clustering.k`.

In `configs/conditional_coverage_configs/evaluate_air_lstm.yaml`, set each
`run_dir` to the real directory containing `log.pkl`. Set `groups_dir` to the
grouping job's `<output_dir>/k<K>`, and choose a new evaluation `output_dir`.
For example, a grouping base of `data/conditional_coverage/air/chronos/groups`
with `clustering.k: 5` requires
`groups_dir: data/conditional_coverage/air/chronos/groups/k5` in the corresponding
evaluation configuration. The requested `nominal_coverage` must match the saved intervals.
Keep `allow_chronological_fallback: true` to support older complete logs without
explicit indices.

## Step 1: build the shared groups

```bash
sbatch sbatch/sbatch_conditional_coverage_experiment/step1_prepare_groups.sbatch \
  --config configs/conditional_coverage_configs/air_lstm.yaml
```

This CPU job trains the independent evaluation encoder for each sequence,
clusters calibration embeddings, and saves the shared grouping artifact. Each
worker handles one sequence at a time with one CPU thread. The number of
workers follows `SLURM_CPUS_PER_TASK`, capped by the number of selected
sequences. It does not rerun the conformal methods. Generate one artifact per
dataset and base-predictor combination and reuse it across method comparisons.

The script requests four CPUs by default. To allocate twelve:

```bash
sbatch --cpus-per-task=12 \
  sbatch/sbatch_conditional_coverage_experiment/step1_prepare_groups.sbatch \
  --config configs/conditional_coverage_configs/air_lstm.yaml
```

With the current 12-sequence Air input, this allows all 12 encoders to train
concurrently. Four CPUs process up to four sequences at a time until all 12
finish. This is one Slurm job with a process pool; it produces one artifact
containing every selected sequence. GPU resources are not requested.

The memory allocation is shared by all worker processes. Adjust `--mem` when
the combined memory needed by the selected number of workers exceeds the
script's default allocation.

## Step 2: evaluate saved results

After step 1 completes successfully:

```bash
sbatch sbatch/sbatch_conditional_coverage_experiment/step2_evaluate.sbatch \
  --config configs/conditional_coverage_configs/evaluate_air_lstm.yaml
```

This CPU job reads the shared groups and existing QR-CP, IQN-CP, and Local-CP
logs. It saves group coverage and sample counts, sequence CACE/WUC, and dataset
summaries. It does not train or rerun any models. If groups already exist, run
only step 2 with the desired saved results.

To queue both jobs with evaluation waiting for successful group generation:

```bash
group_job=$(sbatch --parsable \
  sbatch/sbatch_conditional_coverage_experiment/step1_prepare_groups.sbatch \
  --config configs/conditional_coverage_configs/air_lstm.yaml)

sbatch --dependency="afterok:${group_job%%;*}" \
  sbatch/sbatch_conditional_coverage_experiment/step2_evaluate.sbatch \
  --config configs/conditional_coverage_configs/evaluate_air_lstm.yaml
```

Prepare the evaluation YAML before submitting this pair. The dependency checks
job success; `groups_dir` must still identify the artifact produced by step 1.

## Overrides and dry runs

Both scripts forward additional arguments to their existing Python commands.
For example, append `--dry-run` to audit inputs without creating files, or
`--output-dir /path/to/new/output` to choose a new destination. For step 1, this
overrides the base directory, and the artifact is saved to
`/path/to/new/output/k<K>`; update the evaluation YAML's `groups_dir` accordingly.
Different K subdirectories may coexist under one base. An existing artifact for
the selected K is refused. For step 2, `--output-dir` remains the exact report
directory, and an existing report directory is refused.

A grouping dry run creates no artifact, so it cannot supply groups to a real
evaluation job. It reports the planned worker count without starting training
workers, and its top-level `output_dir` shows the final K subdirectory. An
evaluation dry run requires groups that already exist. Older flat artifacts
remain usable by setting `groups_dir` to their original exact directory.

Without `--config`, step 1 defaults to `air_lstm.yaml` and step 2 defaults to
`evaluate_air_lstm.yaml`. You can also set `CC_GROUP_CONFIG` and
`CC_EVALUATION_CONFIG` before submission. Explicit command-line configuration
paths take precedence over these defaults.

Step 1 always uses CPU workers and takes its worker budget from the Slurm CPU
allocation, overriding a YAML worker count or a forwarded `--cpu-workers`
option. Set `--cpus-per-task` on `sbatch` to change that budget. The previous
`CC_GROUP_DEVICE` override is no longer used. For direct execution outside
Slurm, the Python command supports `--cpu-workers N`; GPU execution is still
available through the Python command with a single worker.

See the [conditional coverage guide](../../conditional_coverage/README.md)
for alignment rules, metrics, and artifact formats.
