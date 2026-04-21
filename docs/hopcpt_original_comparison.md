# HopCPT Implementation Comparison

This note compares the local DSCP HopCPT implementation with the reference
implementation in `HopCPT/`.

## High-impact differences

### 1. Model selection objective

Reference implementation:

- `HopCPT/code/trainer/basetrainer.py` validates the untrained model at epoch 0,
  then selects the best checkpoint from validation epochs.
- With `model_selection: threshold-pi`, selection first minimizes missing
  validation coverage, then compares `PIWidth`.
- Validation metrics are computed on an internal validation split created from
  the calibration data.

DSCP implementation:

- `baselines/hopcpt/run_hopcpt.py` validates every
  `training.validation_epochs`.
- Selection is based on aggregate conformal coverage on `valid` and then
  interval width.
- If a later epoch lowers training loss but drops below target coverage, it will
  not replace an earlier covered checkpoint.

This can make `best_epoch = 5` plausible even when training loss keeps
decreasing.

### 2. Calibration/train/validation split

Reference implementation:

- `CalibTrainerMixin.get_data_loader` splits each calibration sequence into an
  internal train half and validation half.
- The validation split is used only for checkpoint selection inside the
  uncertainty model trainer.
- The fitted model then fills memory from calibration data and performs online
  prediction/evaluation.

DSCP implementation:

- `ConformalPredictionData.prepare_hopcpt_datasets` makes explicit
  train/valid/test splits from the heldout sequence.
- The Hopfield model trains on `heldout_train_context`.
- Validation is the next chronological chunk and includes growing prefix
  memories through `initialize_valid_dataloader`.

These are related but not identical experimental protocols.

### 3. Context features

Reference default HopCPT:

- `ctx_mode: uni_past_multi_step_and_yhat`
- `ctx_past_window: 1` or `2` depending on config.
- Context is `[past y window, current x, current yhat]`.

DSCP current HopCPT:

- `build_hopcpt_context_features` constructs
  `[Y_{t-k}, ..., Y_{t-1}, X_t, yhat_t]`.
- With `y_lags: 1`, this matches the reference one-lag setting.

This part is now close to the reference implementation.

### 4. Residual handling

Reference default HopCPT:

- `predict_abs_eps: True`
- `loss_mode: mse`
- `conf_selection: True`
- `conf_eps_abs: False`
- `conf_quantile_mode: sample`

DSCP current HopCPT:

- `predict_absolute_residual: True`
- `conformal_absolute_residual: False`
- `sampling_num: 1000`

This is conceptually aligned: train MSE on absolute residual magnitudes, but use
signed residuals for conformal interval selection.

### 5. Memory behavior

Reference default HopCPT:

- `online_memory: true`
- `keep_calib_eps: false`
- `eps_mem_size` commonly `400`, `2000`, or `8000` depending on config.
- During test prediction, memory is updated online with newly observed residuals.

DSCP current HopCPT:

- Uses prefix memories for validation/test, so the effective memory also grows
  online.
- Uses `memory_size: 8000`.

The online idea is similar, but DSCP reconstructs prefix dataloaders rather than
using the reference FIFO memory object.

### 6. Sequence batching

Reference default HopCPT:

- `batch_mode: one_ts`
- `batch_size: 2` or `4`
- Each batch item is a whole single time series or subsequence.

Reference mixed variants:

- `batch_mode: naive_mix`
- Large batch sizes.
- Extra data-mixing machinery and, for mixed configs, often
  `conf_quantile_mode: cdf`.

DSCP sequence-batch implementation:

- Trains one shared Hopfield model by stacking independent sequences directly on
  the batch axis.
- This resembles the tensor shape used by the original trainer, but it is not
  the same as the reference `one_ts` training protocol if the original trainer
  creates train/validation halves per series and shuffles DataLoader samples.

This is the biggest remaining structural mismatch for sequence-batch results.

## Recommended comparison experiments

1. **Per-sequence DSCP vs reference-style settings**
   - Use non-sequence-batch DSCP HopCPT first.
   - Match `ctx_past_window/y_lags`, `eps_mem_size`, `predict_abs_eps`, signed
     conformal selection, `n_epochs`, `val_every`, and target alpha.

2. **Checkpoint selection ablation**
   - Save validation coverage, missing coverage, interval width, and validation
     MSE at every validation epoch.
   - Compare:
     - current DSCP threshold coverage/width rule,
     - reference `threshold-pi` rule,
     - validation MSE selection,
     - final epoch.

3. **Batch protocol ablation**
   - `sequence_batch_size: all`, no shuffle.
   - `sequence_batch_size: 4`, shuffle.
   - Reference-style per-series/subsequence batching if implemented later.

4. **Quantile mode ablation**
   - Current DSCP uses sampling.
   - Add an exact weighted CDF quantile path to remove Monte Carlo noise and
     compare with the reference `conf_quantile_mode: cdf`.

## Most likely causes of the observed discrepancy

1. Selection by coverage/width can prefer epoch 5 even when MSE improves.
2. Sequence-batch DSCP is not a faithful port of the reference trainer's
   `one_ts` DataLoader protocol.
3. The previous sequence-batch config accidentally compared a Chronos-named run
   against LR data and different split/normalization/training settings.
4. Monte Carlo quantile sampling adds noise to coverage/width based selection.
