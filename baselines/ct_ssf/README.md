# CT-SSF on Chronos and LSTM

These are the **CT-SSF Chronos and LSTM adaptations** (`ct_ssf_chronos_adapted`
and `ct_ssf_lstm_adapted`). Read
[UPSTREAM.md](UPSTREAM.md) before reporting results: the official attention path
is incomplete, and this runner documents its causal replacement and its use of
uniform-gradient latent inversion.

Implementation version 2 aligns attention with the repaired local `CT-SSF/`
clone: learn similarity between **feature displacement vectors** `u-z` using a
KL loss. It replaces the earlier scalar log-score regression. The Chronos
adaptation retains causal memory, chronological validation, a frozen pretrained
predictor, and its exported median head. No import from the clone is required.

Existing feature/head exports remain compatible. Rerun the CT-SSF stage to fit
the new attention; old attention checkpoints/results describe version 1 and
must not be mixed with version 2. The three existing sbatch commands still work.

## Run on an existing saved LSTM

The existing `LSTMPredictor.save(..., save_model=True)` output is sufficient.
Each dataset needs the matching `lstm_<data_type>_data.pkl` and
`lstm_<data_type>_model.pt`. CT-SSF loads the checkpoint and its normalization,
rebuilds rolling one-step windows from the saved training/held-out observations,
and extracts `z_t = model.encode(window_t)[:, -1, :]`. Window t ends at t-1;
neither the current target nor its current covariates enter its feature.
Each target has its own window and feature. The base model is frozen and is
not retrained here.

Features are computed in batches during the CT-SSF run. No new export flag or
change to the original prediction files is needed. The runner verifies that the
saved output layer and normalization reconstruct every saved forecast. A
mismatched model/forecast pair fails explicitly. Results and attention checkpoints
record fingerprints of both the head and the full LSTM checkpoint.

The LSTM output head is linear: `g(z) = w z + b`. With radius r, its exact
normalized-output range is `[g(z)-r*sum(abs(w)), g(z)+r*sum(abs(w))]`.
`bound_method: linear_exact` uses that range and restores the original response
scale, with a small outward numerical margin. Feature inversion still uses the
same finite-step SGD/Adam implementation as the Chronos adapter; exact head
bounds do not make that inversion exact or guarantee target coverage.

The presets use the directories produced by the existing save-model jobs:

| Dataset | Directory | File prefix |
| --- | --- | --- |
| Air | `data/air-10_prediction/lstm_model_saved/` | `lstm_air-10` |
| Solar | `data/solar_prediction/lstm_model_saved/` | `lstm_nsdb-60m` |
| Sapflux | `data/sapflux-solo3-large/lstm_model_saved/` | `lstm_sapflux-solo3-large` |

If these files already exist, submit only CT-SSF:

```bash
for dataset in air solar sapflux; do
  sbatch "sbatch/sbatch_run_ct_ssf/run_ct_ssf_lstm_${dataset}.sbatch"
done
```

If the checkpoint has not been saved, run the existing base-predictor jobs first:

```bash
for dataset in air solar sapflux; do
  job=$(sbatch --parsable "sbatch/sbatch_run_base_predictor/run_lstm_${dataset}_save_model.sbatch")
  sbatch --dependency="afterok:${job%%;*}" \
    "sbatch/sbatch_run_ct_ssf/run_ct_ssf_lstm_${dataset}.sbatch"
done
```

The new CT-SSF jobs default to the `dscp` environment (which includes OmegaConf);
override with `CTSSF_ENV` if necessary. They accept the same `DSCP_RUNPATH`,
`CTSSF_SEED`, and extra CLI arguments as the Chronos jobs. They do not need the
Chronos package or Auto-LiRPA.

For an interactive run or config inspection:

```bash
conda activate dscp
python -m sbatch.sbatch_run_ct_ssf.run_ct_ssf --dataset air --base-predictor lstm --dry-run
python -m sbatch.sbatch_run_ct_ssf.run_ct_ssf --dataset air --base-predictor lstm --device cuda
```

Use `--device cpu` for CPU execution. All three datasets have dedicated CPU Slurm jobs:

```bash
sbatch sbatch/sbatch_run_ct_ssf/run_ct_ssf_lstm_air_cpu.sbatch
sbatch sbatch/sbatch_run_ct_ssf/run_ct_ssf_lstm_solar_cpu.sbatch
sbatch sbatch/sbatch_run_ct_ssf/run_ct_ssf_lstm_sapflux_cpu.sbatch
```

Each job requests 12 CPU cores and 48 GB of memory, with no GPU allocation, and
runs the saved LSTM/CT-SSF pipeline in the `dscp` environment. These jobs can load
checkpoints originally trained on GPU. Adding `--device cpu` to a GPU sbatch
script changes computation but does not remove that script's GPU allocation;
use the dedicated CPU script when submitting a CPU job.

Override paired file locations with
`--data-path /path/lstm_air-10_data.pkl --model-path /path/lstm_air-10_model.pt`.
An explicit LSTM YAML selects the predictor via `base_predictor: lstm`.
Results are written to `results/ct_ssf/<dataset>/lstm/seed_<seed>/<timestamp>/`.

The 40%/10%/16%/34% CT-SSF splits are applied to the LSTM's **held-out suffix**,
after its original base-model training prefix. Attention fitting, validation,
rolling calibration, inversion diagnostics, and result fields otherwise follow
the shared protocol described below. Compare other LSTM baselines using the
same saved forecasts and held-out `target_indices`. When comparing different
base predictors, align `raw_target_indices` because their held-out suffixes can
start at different raw timestamps.

## Run on Chronos for all three datasets

From the repository root on the cluster, first add OmegaConf to an existing
Chronos environment (new environments include it in `envs/env-chronos.yml`):

```bash
conda activate chronos
python -m pip install omegaconf==2.3.0
```

Submit feature export, then schedule calibration after that export succeeds:

```bash
for dataset in air solar sapflux; do
  job=$(sbatch --parsable "sbatch/sbatch_run_base_predictor/run_chronos_${dataset}_ctssf.sbatch")
  sbatch --dependency="afterok:${job%%;*}" \
    "sbatch/sbatch_run_ct_ssf/run_ct_ssf_${dataset}.sbatch"
done
```

Both stages use the `chronos` environment. Jobs inherit `DSCP_RUNPATH` (default:
the submission directory), `CTSSF_ENV` (default: `chronos`), and `CTSSF_SEED`
(default: 2026). Export jobs also accept `CHRONOS_REVISION`, which can pin a
Hugging Face model commit. Slurm account/GPU directives follow the existing
Chronos jobs. These scripts are provided for submission; no jobs are submitted
by installing or testing this code.

Inspect a preset without running or writing anything:

```bash
python -m sbatch.sbatch_run_ct_ssf.run_ct_ssf --dataset air --dry-run
```

Run locally or interactively after feature export:

```bash
python -m sbatch.sbatch_run_ct_ssf.run_ct_ssf --dataset air --device cuda
python -m sbatch.sbatch_run_ct_ssf.run_ct_ssf --dataset solar --device cuda
python -m sbatch.sbatch_run_ct_ssf.run_ct_ssf --dataset sapflux --device cuda
```

`--device cpu`, `--seed`, `--num-threads`, `--data-path`, `--head-path`, and
`--output-dir` override a preset. A positional YAML path may replace `--dataset`.
For example, manually exporting Air is:

```bash
python -m sbatch.sbatch_run_base_predictor.run_chronos air-10 \
  --data-dir data/bejing_air_quality \
  --save-dir data/air-10_prediction/chronos_ctssf \
  --window-length 100 --prediction-length 10 --device cuda --save-ctssf
```

## Chronos saved artifacts and compatibility

| Dataset | Feature/head directory | File prefix |
| --- | --- | --- |
| Air | `data/air-10_prediction/chronos_ctssf/` | `chronos_air-10` |
| Solar | `data/solar_prediction/chronos_ctssf/` | `chronos_nsdb-60m` |
| Sapflux | `data/sapflux-solo3-large/chronos_ctssf/` | `chronos_sapflux-solo3-large` |

`<prefix>_data.pkl` retains the ordinary per-series `heldout_x`, `heldout_y`,
and `heldout_predictions`. The optional `ctssf` entry contains:

- Deduplicated forecast-patch features, and each target's feature/output indices.
- Context location/scale, forecast origin, lead, and raw target index.
- Artifact format version and the paired head's SHA-256 fingerprint.

`<prefix>_ctssf_head.pt` contains the frozen median head's configuration and
weights, package/model provenance, and fingerprint. It is sufficient for latent
optimization and interval construction without loading the full Chronos model.
It is not a full forecasting checkpoint. A normal run without `--save-ctssf`
keeps the original output schema. Existing prediction-only files cannot recover
these features; regenerate them with the flag.

Every export verifies prediction reconstruction. The baseline revalidates that
reconstruction, the paired weights, and all time/lead mappings. Unsupported
head architectures, non-float32 weights, or autoregressive long-horizon
unrolling fail explicitly. Tested against `chronos-forecasting==2.2.2`.

## Comparison protocol

The saved forecast suffix is split chronologically into 40% attention training,
10% epoch validation, 16% calibration, and 34% test. Cumulative boundaries are
floored: test starts at `floor(0.66 * N)`. Training affects only attention, not
Chronos. The calibration window uses up to 500 recent eligible observations.
Set `rolling_calibration: false` for a fixed calibration pool.

Each observation is evaluated as a scalar target, as in the current pipeline;
this is not simultaneous coverage of the 10-step forecast vector. Base
predictions/features are fixed at each block origin, while conformal calibration
may update after each target is observed. Thus these intervals are issued
prequentially, not all ten at the block origin. No future covariates are used.

The presets evaluate 90% and 80% coverage. Pair labels indicate nominal
central coverage; the nonlinear output band need not have equal empirical tail
errors. Tuning must use training/validation data, leaving the final test suffix
untouched. Preset inversion/attention parameters are initial settings, not
dataset-specific tuned results.

The new jobs write separate `chronos_ctssf` directories. For fair comparisons,
point the other methods at **these same forecast artifacts**, or first verify
that existing Chronos forecasts/targets and model revision agree. Existing
baseline readers ignore the extra per-series `ctssf` entry. Match nominal
coverage and align by `target_indices`; some other baselines remove warmup rows.

Results go to `results/ct_ssf/<dataset>/chronos/seed_<seed>/<timestamp>/`:

- `launch_config.yaml`, `resolved_config.yaml`.
- Standard `log.pkl` and `summary_results.pkl` with original-scale endpoints,
  coverage, width, Winkler score, predictions, and original held-out indices.
- One saved attention checkpoint per series, including training normalization.
- Per-series provenance, split boundaries, inversion errors, forecast leads,
  effective calibration sizes, and counts of infinite intervals.

`avg_coverage` always measures the **observed output** inside the interval.
`avg_feature_coverage` instead measures whether the inverted surrogate is inside
the latent ball; `avg_surrogate_output_coverage` checks its reconstructed output
against the band. The latter two are diagnostics and do not establish coverage
of the observed target. All three can differ when inversion is inaccurate.

`inverse_diagnostics` reports absolute model-scale errors, aggregate relative
squared reconstruction errors in both model and original response units, and
the fraction of rows with response relative squared error below 1%. The relative
errors divide the sum of squared reconstruction errors by the sum of squared
targets in each fold. For the row criterion, zero targets pass only when their
reconstruction is exact. The 1% check diagnoses inversion; it does not guarantee
90% output coverage. `inverse_settings` records the fixed optimizer and step
count. Choose these on training/validation data, not on reported test coverage;
the exploratory wind step sweep is not a Chronos preset.

Independent inverse problems are batched for speed, including test rows, but
each interval's quantile only reads already revealed scores. Test labels are
never used to train/select attention. Infinite intervals are preserved; their
width/Winkler means are infinite and their across-series standard deviations
are undefined (`NaN`). The runner does not replace them with a finite cap.

## Validation

```bash
python -m pytest tests/test_ct_ssf.py tests/test_ct_ssf_lstm.py tests/test_chronos_predictor_context.py \
  tests/test_chronos_ctssf_integration.py -q
```

Tests check numerical bound containment, finite-sample weighted quantiles,
frozen-head inversion, independence from execution batch size, artifact
alignment, chronology, absence of future-label leakage, and standard result
files. They also check learning from displacement direction when scalar scores
are identical, attention checkpoint restoration, and the distinction between
feature and output coverage. The integration tests instantiate a tiny real Chronos-2 model with random
weights; they require the Chronos package but download no pretrained weights.
The LSTM tests train tiny predictors locally, save/reload the existing checkpoint
format, and exercise the full CT-SSF CLI. They check prediction reconstruction,
feature alignment, unchanged base weights/files, future-observation isolation,
normalization (including constant covariates), and exact linear-head bounds.
