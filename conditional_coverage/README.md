# Group conditional coverage

This module builds one reusable grouping artifact for conditional coverage
experiments. It trains a fresh median forecasting encoder for each sequence,
clusters calibration representations, and saves the group membership of shared
test observations. A separate evaluation command reads saved QR-CP, IQN-CP,
and Local-CP results and reports coverage within those same groups without
training either the conformal models or the grouping encoder again.

## Run a preset

Use the environment defined in `envs/env-dscp.yml`. From the repository root,
first audit the source data, comparison configurations, and exact split indices:

```bash
python -m conditional_coverage.prepare_groups --config configs/conditional_coverage_configs/air_lstm.yaml --dry-run
```

The dry run loads the saved forecast artifact and validates all selected
sequences. It prints calibration, fitting, validation, and test context counts,
each method's boundary, any gap between the shared partitions, and the planned
worker count. It does not train a model or write output files.

Build and save the groups with:

```bash
python -m conditional_coverage.prepare_groups --config configs/conditional_coverage_configs/air_lstm.yaml
```

The default device is CPU, with one worker. To process up to four sequences
concurrently, using one CPU thread per worker:

```bash
python -m conditional_coverage.prepare_groups --config configs/conditional_coverage_configs/air_lstm.yaml --cpu-workers 4
```

`--cpu-workers` selects CPU execution and sets `num_threads` to 1. Each worker
trains one sequence's encoder, extracts its embeddings, and fits its K-means
partition before taking another sequence. The worker count is capped by the
number of selected sequences and, under Slurm, the allocated CPUs. All workers
contribute to one final artifact, saved after every sequence succeeds.

These settings can also be supplied in the grouping YAML:

```yaml
device: cpu
cpu_workers: 4
num_threads: 1
```

`cpu_workers` defaults to 1. Configuring more than one worker requires
`device: cpu` and `num_threads: 1`. Direct GPU execution remains available with
one worker. To select a GPU and a new output directory:

```bash
python -m conditional_coverage.prepare_groups --config configs/conditional_coverage_configs/air_lstm.yaml --device cuda:0 --output-dir results/conditional_coverage/air/lstm/experiment_02
```

An existing output directory is refused. Choose a new directory for a changed
experiment or another seed. The completed artifact is published only after all
selected sequences have trained and passed artifact validation.

Nine presets are provided under `configs/conditional_coverage_configs/`:

| Dataset | Linear regression | LSTM | Chronos |
| --- | --- | --- | --- |
| Air | `air_lr.yaml` | `air_lstm.yaml` | `air_chronos.yaml` |
| Solar | `solar_lr.yaml` | `solar_lstm.yaml` | `solar_chronos.yaml` |
| Sapflux | `sapflux_lr.yaml` | `sapflux_lstm.yaml` | `sapflux_chronos.yaml` |

Each preset includes six comparisons: RNN and Transformer encoders for each
of QR-CP, IQN-CP, and Local-CP. Additional
variants, including fixed-quantile IQN, can be added to `comparisons` using their
own configuration or saved run. The comparison list determines the shared
partitions and is recorded in the artifact.

## Run through Slurm

Use `sbatch/sbatch_conditional_coverage_experiment/step1_prepare_groups.sbatch`
to generate groups on CPUs, followed by
`sbatch/sbatch_conditional_coverage_experiment/step2_evaluate.sbatch` to evaluate
saved method results. Step 1 uses one worker per allocated CPU, capped by the
number of sequences, and one CPU thread per worker. For example, submit it with
`sbatch --cpus-per-task=12 ...` to process up to 12 sequences concurrently.
Both accept `--config` and forward the existing command-line options; step 1
enforces CPU execution and takes its worker budget from the Slurm allocation.
The [Slurm workflow guide](../sbatch/sbatch_conditional_coverage_experiment/README.md)
includes submission commands and an `afterok` dependency that starts evaluation
only after group generation succeeds.

## Observations and chronological partitions

The source is a saved base-predictor `*_data.pkl` artifact. A sequence supplies
`heldout_x` and `heldout_y`; predictions and residuals are not encoder inputs or
training labels. `heldout_y[j]` is the observed response at index `j` within
that artifact's held-out timeline. Only scalar, one-step targets are supported.

For each sequence, the full pre-test prefixes of all selected methods are
intersected. This normally covers approximately the first 66% of the held-out
sequence, including observations that individual methods call training,
validation, or calibration. The original test suffixes are also intersected,
using the exact rounding rules of the selected methods.

For example, if methods start testing at indices 66 and 67 in a length-101
sequence, shared calibration contains indices `0..65` and shared test contains
`67..100`. Index 66 is excluded from encoder fitting and scored test membership,
but can be previously observed history for a later test context. Prediction
exclusions in method logs do not redefine either shared partition.

Comparisons must reference the same forecast artifact, or byte-identical copies
of it. LR, LSTM, and Chronos artifacts can have different held-out timelines;
combine methods within one verified source timeline. Cross-forecaster timeline
mapping is outside this stage.

## Encoder and clustering protocol

For a target at index `j`, a context contains the preceding `W` rows of observed
covariates and responses, `concat(x[j-W:j], y[j-W:j])`. The encoder predicts the
median of the observed response `y[j]`. The current target never enters its own
context.

Defaults are a one-layer LSTM with 64 hidden units, a 100-observation context,
and one linear median head. Training minimizes `0.5 * abs(y - prediction)`,
the pinball loss at quantile 0.5. Training uses Adam with learning rate 0.001,
batch size 64, at most 50 epochs, and early-stopping patience 5.

The earliest 80% of the shared calibration prefix supplies fitting data and the
remaining chronological 20% selects the checkpoint. The first `W` observations
are context warmup, so all fitting contexts are wholly inside the prefix.
Covariate and target normalization statistics are fitted on the fitting prefix
only; constant columns use scale 1. The selected checkpoint and preprocessing
are frozen. There is no additional refit after checkpoint selection.

The final hidden state is extracted for every eligible calibration context and
every shared test context, then L2-normalized. Earlier observed test outcomes
may enter later contexts; no test outcome trains the encoder or changes its
normalization. Zero or nonfinite embeddings are rejected.

Each run fits one K-means partition per sequence using the frozen calibration
embeddings, including the chronological checkpoint-validation portion. Set the
number of groups with a single positive integer `clustering.k` in the YAML:

```yaml
clustering:
  k: 5
  n_init: 10
```

The default is five groups. Change `k` to the desired number; the builder fits
only that partition. `n_init` controls the number of K-means initializations for
the selected K and defaults to 10. The former `clustering.ks` list is no longer
accepted. Test embeddings are assigned to the nearest fitted Euclidean centroid;
centroids are not renormalized.

One encoder and one clustering model are fitted per sequence. A root seed of 2026
generates stable, separate encoder and clustering seeds from each sequence's
identifier. Reordering sequences does not change their seeds. The artifact
records seeds and package versions; exact numerical reproducibility should be
checked on the same hardware and software environment.

These groups define a learned partition for later coverage diagnostics. Median
prediction is not a test of every possible coverage-relevant regime.

## Configuration paths and existing runs

The supplied YAMLs set `path_base: ../..`, resolved relative to their own
directory, so data and comparison paths resolve from the repository root.
An explicit CLI `--output-dir` is resolved relative to the current directory.

A comparison can read a method configuration without requiring a completed run:

```yaml
comparisons:
  - name: qr_rnn
    method: qr_cp
    config_path: configs/qr_cp_configs/qr_rnn_lstm_air_config.yaml
```

To verify actual saved runs, replace `config_path` with `run_dir`:

```yaml
comparisons:
  - name: qr_rnn_saved
    method: qr_cp
    run_dir: results/qr_cp/air/lstm/rnn/independent/my_run
```

Each run must contain `resolved_config.yaml` and `log.pkl`. The builder checks
available original indices, exclusion masks, boundaries, and logged target
values against the source artifact. Ambiguous or inconsistent alignments fail.
When runs contain different sequence subsets, only sequences present in every
run are used; skipped sequences are listed in the audit. An explicit `sequences`
list requires each requested sequence to be present in all runs.

Method configurations ordinarily resolve their `data.data_path` from the
repository root. A comparison's optional `source_root` sets a different root
for that method's relative source path. For a relocated run with an obsolete
absolute data path, `source_data_path` can explicitly point to its local source
artifact:

```yaml
comparisons:
  - name: qr_rnn_relocated
    method: qr_cp
    run_dir: downloaded_runs/qr_rnn
    source_data_path: data/air-10_prediction/lstm/lstm_air-10_data.pkl
```

Source, configuration, and supplied log fingerprints are recorded. Use
`source_data_path` only to identify the same source timeline after relocation.

Grouping does not use interval endpoints or nominal coverage levels. Therefore
the current 80% presets can identify splits for this stage. Evaluating nominal
90% coverage requires saved 90% intervals from each selected method.

## Saved artifact

The default output is
`results/conditional_coverage/{dataset}/{predictor}/groups/`:

| File | Contents |
| --- | --- |
| `manifest.json` | Selected `k`, source and comparison fingerprints, shared-split audit, protocol, package versions, sequence checkpoint references, and file checksums |
| `resolved_config.yaml` | Complete resolved evaluator configuration |
| `groups.pkl` | Per-sequence indices, normalization, training history, centroids, labels, and counts for the selected K |
| `encoders/<identifier-hash>.pt` | Per-sequence model state, architecture, scaler, seed, and selected epoch |

`groups.pkl` stores group membership as indices and label arrays, rather than
duplicating the original test samples. Each sequence record contains:

- `calibration_indices`: the full shared pre-test prefix.
- `calibration_target_indices`: eligible calibration targets after context warmup.
- `fit_indices` and `validation_indices`: encoder supervision/checkpoint-selection targets.
- `test_indices`: the shared original test observations, before prediction exclusions.
- `excluded_boundary_indices` and `member_boundaries`: the rounding gap and each method's split.
- `scaler` and `training`: normalization statistics and encoder training metadata.
- `groups[K]`: a single entry for the configured K containing calibration/test labels and counts, centroids, clustering seed, `n_init`, and inertia.
- `calibration_embeddings` and `test_embeddings`: cached unit representations when `cache_embeddings: true`.

Labels are zero-based and local to each sequence. An empty test group is
retained with count zero. Degenerate calibration clustering that cannot produce
K nonempty groups raises an error.

Load a completed artifact through the checksum-validating reader:

```python
from conditional_coverage.artifacts import load_group_artifact

manifest, sequences = load_group_artifact(
    "results/conditional_coverage/air/lstm/groups"
)
sequence_key = next(iter(sequences))
record = sequences[sequence_key]
target_indices = record["test_indices"]
group_ids = record["groups"][manifest["k"]]["test_labels"]
group_0_target_indices = target_indices[group_ids == 0]
```

Use artifacts generated by the experiment workflow: the saved source data,
groups, and model checkpoints use Python pickle-based formats. Conditional
coverage evaluation joins each method's results to these exact target
indices without retraining the grouping encoder.

New QR-CP, IQN-CP, and Local-CP runs save
`log[sequence_key]["evaluation_results"][quantile_pair]["target_indices"]`.
These are zero-based positions in that sequence's original `heldout_y`, aligned
one-to-one with `coverage`, `target_y`, and the saved interval endpoints.
QR-CP and IQN-CP modes that exclude quantile crossings save only retained
indices in each result; their sequence metadata keeps the full test indices
and `valid_prediction_mask`. Join results to shared groups by sequence key and
target index. Existing result files are not changed by this update.

## Evaluate saved method results

The evaluation stage supports only `qr_cp`, `iqn_cp`, and `local_cp`. Multiple
runs of any supported method can be compared, including different encoders,
heads, or hyperparameters. The grouping presets and evaluation example both
contain the six RNN/Transformer combinations of these three methods. The
evaluator uses the saved shared partition for every selected run.

Edit `configs/conditional_coverage_configs/evaluate_example.yaml`: replace each
`REPLACE_WITH_RUN_DIRECTORY` path with a completed run directory containing
`log.pkl`, and set `groups_dir` to the existing grouping artifact. Remove any
run entries you do not want to compare. Then audit the inputs:

```bash
python -m conditional_coverage.evaluate --config configs/conditional_coverage_configs/evaluate_example.yaml --dry-run
```

The dry run validates inputs and computes the report without writing output.
Remove `--dry-run` to save it. The command reads saved coverage indicators,
targets, and indices; it does not load or run the conformal model checkpoints.
Choose a new output directory with `--output-dir` when needed. Existing output
directories are refused.

The evaluation configuration is separate from the grouping configuration:

| Option | Meaning |
| --- | --- |
| `path_base` | Base for relative paths, resolved from the evaluation YAML directory |
| `groups_dir` | Directory containing the saved grouping manifest and `groups.pkl` |
| `output_dir` | New directory for evaluation reports |
| `nominal_coverage` | Requested interval coverage, default `0.9` |
| `allow_chronological_fallback` | Allow inference of target indices for older logs; default `false`, enabled in the example |
| `data_path` | Optional local path to the same source forecast artifact after relocation; its checksum must match the grouping manifest |
| `sequences` | Optional explicit list of sequence identifiers, all required in every selected run and the grouping artifact |
| `runs` | Runs to evaluate, each with a unique `name`, supported `method`, and `run_dir` |

Each run can additionally specify `config_path`, `quantile_pair`, or its own
`allow_chronological_fallback`. If the run has `resolved_config.yaml`, the
evaluator uses that saved configuration to establish its exact test boundary.
For an older run without that file, `config_path` can supply the configuration
used for that run. Use its original split settings, rather than a newer
configuration with different ratios. `quantile_pair: [0.05, 0.95]` selects a
particular interval when the log contains multiple pairs with the requested
nominal coverage.

By default, evaluation uses sequences present in the grouping artifact and
every selected run, and records skipped sequences in the audit. The same
saved group partition and shared test indices are used for all runs. The
number of groups comes from the artifact's single saved `k`; changing K
requires building a new grouping artifact.

The nominal coverage must match the difference between the saved interval's
quantile levels. A log containing only 80% intervals cannot be evaluated as a
90% interval result. Set `nominal_coverage: 0.8` to evaluate those existing
intervals at their own nominal level, or supply runs that already saved the
desired 90% intervals.

### Older logs without indices

Enable chronological inference in YAML:

```yaml
allow_chronological_fallback: true
```

The CLI flag `--allow-chronological-fallback` also enables this option. Explicit
per-result `target_indices` take precedence. If they are absent, the evaluator
first tries original indices in sequence metadata and applies any saved
`valid_prediction_mask` to preserve excluded positions.

Only when explicit index information is unavailable does the enabled fallback
treat saved rows as chronological. With a method configuration, rows are
aligned to that method's exact test suffix, including its split-rounding rule.
Without one, the evaluator assumes the rows cover the final chronological
suffix of the original held-out sequence. A saved validity mask identifies the
full timeline and retains any gaps from excluded predictions.

In every case, saved `target_y` values must match the original observed targets
at the inferred or explicit indices. The report records how each sequence was
aligned. Malformed explicit indices, inconsistent lengths, and known
exclusions without enough information to locate their gaps are rejected.
The fallback assumes a complete chronological suffix when no configuration or
exclusion metadata is available; it cannot reconstruct arbitrary omitted
observations from row order alone. Older complete logs need no model rerun or
rewrite.

When explicit retained indices are available but the original test size is
unknown, the alignment audit leaves `full_test_count` and `excluded_count`
null. Group-level `missing_count` remains exact because shared membership is
known.

### Coverage metrics and aggregation

For each sequence and group, coverage is the mean of saved coverage indicators
whose target indices belong to that group. Rows outside the shared test
intersection are omitted, including any extra boundary observation produced
by split rounding. Every group reports its original `group_size`,
`evaluated_count`, `missing_count`, and `covered_count`.

Saved prediction exclusions remain excluded from coverage denominators. They
do not become covered observations, and they do not change group membership.
Thus a group's reported coverage describes its available predictions; use the
missing counts when comparing methods with different exclusion rates.

For nominal coverage `p`, each sequence reports:

- `cace`: the equal-weight mean of `abs(group_coverage - p)` across groups
  having evaluated predictions.
- `wuc`: the largest `max(p - group_coverage, 0)` across those groups.
- `coverage`: pooled coverage over evaluated observations in that sequence.
- `evaluated_group_count` and `empty_group_count`: the actual group denominator
  and number of groups without evaluated predictions.

When all K groups have predictions, CACE uses the experimental `1 / K`
denominator. An empty group has null coverage and error values and is omitted
from the sequence's CACE and WUC. A sequence with no evaluated predictions has
null metrics and is counted as unavailable in the summary.

Dataset summaries average each sequence's CACE and WUC with equal sequence
weight, using evaluable sequences. They also report `worst_sequence_wuc`,
`mean_sequence_coverage`, and `pooled_coverage`. Cluster labels are local to
each sequence; the evaluator never averages coverage for an arbitrary label
across different sequences.

### Evaluation output

| File | Contents |
| --- | --- |
| `group_metrics.csv` | One row per run, sequence, and group, with coverage, errors, and sample counts |
| `sequence_metrics.csv` | Per-sequence CACE, WUC, coverage, availability, and counts |
| `summary.csv` | Dataset-level metrics for each named run |
| `evaluation.json` | Complete results and alignment/provenance audit |
| `resolved_config.yaml` | Resolved evaluation settings and run paths |

JSON uses `null` for unavailable metrics; CSV leaves those cells empty. The
input grouping artifact and all method logs remain unchanged.
