# CT-SSF on Chronos

This is the **CT-SSF Chronos adaptation** (`ct_ssf_chronos_adapted`). Read
[UPSTREAM.md](UPSTREAM.md) before reporting results: the official attention path
is incomplete, and this runner documents its causal replacement and its use of
uniform-gradient latent inversion.

## Run all three datasets

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

## Saved artifacts and compatibility

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

Independent inverse problems are batched for speed, including test rows, but
each interval's quantile only reads already revealed scores. Test labels are
never used to train/select attention. Infinite intervals are preserved; their
width/Winkler means are infinite and their across-series standard deviations
are undefined (`NaN`). The runner does not replace them with a finite cap.

## Validation

```bash
python -m pytest tests/test_ct_ssf.py tests/test_chronos_predictor_context.py \
  tests/test_chronos_ctssf_integration.py -q
```

Tests check numerical bound containment, finite-sample weighted quantiles,
frozen-head inversion, independence from execution batch size, artifact
alignment, chronology, absence of future-label leakage, and standard result
files. The integration tests instantiate a tiny real Chronos-2 model with random
weights; they require the Chronos package but download no pretrained weights.
