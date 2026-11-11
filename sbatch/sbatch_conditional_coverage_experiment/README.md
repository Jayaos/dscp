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

In `configs/conditional_coverage_configs/evaluate_example.yaml`, replace each
run-directory placeholder with the real directory containing `log.pkl`.
Set `groups_dir` to the grouping job's `output_dir`, and choose a new evaluation
`output_dir`. The requested `nominal_coverage` must match the saved intervals.
Keep `allow_chronological_fallback: true` to support older complete logs without
explicit indices.

## Step 1: build the shared groups

```bash
sbatch sbatch/sbatch_conditional_coverage_experiment/step1_prepare_groups.sbatch \
  --config configs/conditional_coverage_configs/air_lstm.yaml
```

This GPU job trains the independent evaluation encoder for each sequence,
clusters calibration embeddings, and saves the shared grouping artifact. It
does not rerun the conformal methods. Generate one artifact per dataset and
base-predictor combination and reuse it across the method comparisons.

## Step 2: evaluate saved results

After step 1 completes successfully:

```bash
sbatch sbatch/sbatch_conditional_coverage_experiment/step2_evaluate.sbatch \
  --config configs/conditional_coverage_configs/evaluate_example.yaml
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
  --config configs/conditional_coverage_configs/evaluate_example.yaml
```

Prepare the evaluation YAML before submitting this pair. The dependency checks
job success; `groups_dir` must still identify the artifact produced by step 1.

## Overrides and dry runs

Both scripts forward additional arguments to their existing Python commands.
For example, append `--dry-run` to audit inputs without creating files, or
`--output-dir /path/to/new/output` to choose a new destination. If you override
the grouping destination, update the evaluation YAML's `groups_dir` too.
Existing output directories are refused.

A grouping dry run creates no artifact, so it cannot supply groups to a real
evaluation job. An evaluation dry run requires groups that already exist.

Without `--config`, step 1 defaults to `air_lstm.yaml` and step 2 defaults to
`evaluate_example.yaml`. You can also set `CC_GROUP_CONFIG` and
`CC_EVALUATION_CONFIG` before submission. `CC_GROUP_DEVICE` defaults to `cuda:0`;
step 1 accepts `--device` to override it. Explicit command-line options take
precedence over these defaults.

See the [conditional coverage guide](../../conditional_coverage/README.md)
for alignment rules, metrics, and artifact formats.
