# CT-SSF provenance and implementation choices

This experiment is labeled **`ct_ssf_chronos_adapted`**, not an exact reproduction
of the authors' reported CT-SSF results. It implements a usable Chronos semantic
feature baseline using the authors' latent inversion, weighted conformal
quantile, and output-band construction. It makes the missing attention path
explicit and keeps test labels out of model fitting and interval construction.

## Sources inspected

- [CT-SSF repository at 28320c61374c93746db8170b641c29cbf313d7bc](https://github.com/baiting0522/CT-SSF/tree/28320c61374c93746db8170b641c29cbf313d7bc).
- [Earlier latent optimizer at 85843cbc4d54470881b2ee958d692c5d47448375](https://github.com/baiting0522/CT-SSF/blob/85843cbc4d54470881b2ee958d692c5d47448375/conformal/icp.py),
  especially `FeatRegressorNc.inv_g`, `score`, and `predict`.
- [Chronos-2 model at v2.2.2](https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos2/model.py),
  [residual head](https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos2/layers.py),
  [pipeline](https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos2/pipeline.py),
  and [instance normalization](https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos_bolt.py).
  Chronos is Apache-2.0 licensed. This adapter uses its exported weights and
  independently expresses its Linear/ReLU/Linear-plus-skip median head.

No upstream repository is vendored or imported by this baseline.

## Why this is an adaptation

The inspected CT-SSF HEAD is not executable as published. `main.py` references
undefined `weights` and passes a keyword unsupported by the helper adapter.
`FeatRegressorNc.inv_g` has an incompatible signature at its callers and updates
model weights rather than surrogate features. Its prediction path calls missing
`compute_attention_weights`, refers to undefined `x_calib`, and uses an
inconsistent weighted-quantile call. Earlier history contains the intended
feature optimizer but its attention path is commented out and `default_loss`
is imported without a matching definition.

The paper's Algorithm 1 also writes test outcomes into its weight/feature update.
A forecasting experiment cannot use an unrevealed test outcome to construct its
own interval. The paper does not provide enough executable detail to recover a
unique causal attention objective or its adaptive gradient weights.

| Component | This implementation |
| --- | --- |
| Base predictor | Frozen pretrained Chronos-2, with the existing 100-context, 10-step rolling forecast protocol and past-only covariates. No base-model training. |
| Semantic feature | The actual forecast-patch input to `output_patch_embedding`, captured during `predict_df`. Each scalar lead selects its own median output coordinate. |
| Head | ReLU residual head restricted to median outputs, followed by the saved context's inverse normalization. The Transformer is outside the perturbed subnetwork. |
| Surrogate feature / score | Optimize each feature independently against the observed target using the frozen head; take `max(abs(z_surrogate - z_forecast))`. This follows the earlier upstream optimizer. |
| Inversion loss | Squared error in the head's output scale, before inverse `sinh` and affine normalization. These transforms are invertible, so an exact solution has the same target constraint. The finite optimization path differs from raw-response loss. |
| Inversion weights | Uniform. The paper's Eq. (7) adaptive gradient weighting is **not reconstructed**. No claim of exact CT-SSF reproduction is made. |
| Optimizer settings | Configurable SGD/Adam, fixed steps; presets use Adam, 200 steps, learning rate 0.01. These are starting settings, not tuned paper parameters. Sum of independent per-row losses avoids changing SGD's learning rate with execution batch size. |
| Relevance learning | DSCP repair: shared two-layer query/key projection, cosine attention, trained to predict standardized `log1p(score)` from earlier scores. Inputs are forecast features and output-position one-hot encoding. No current outcome or current residual enters a query. |
| Training / validation | Chronological 40%/10%; validation chooses the attention epoch. Feature normalization and objective normalization use the training prefix only. Attention is frozen before the next 16% calibration fold. |
| Weighted quantile | Eq. (8), with calibration weights and a test weight on the **same scale**. The test atom is at positive infinity. Low support can therefore yield an infinite interval. |
| Band estimation | Analytic IBP or single-hidden-layer CROWN for an L-infinity ball. CROWN is intersected with IBP. Monotone `sinh`/affine inversion maps bounds to original response units. This replaces Auto-LiRPA for this supported head architecture only. |
| Evaluation | Final 34%, with a sliding window of the latest 500 eligible scores. Test score t can enter the pool only for intervals after t. Uniform weighting is available as a FeatureCP-style ablation. |

The attention objective uses a softmax over the preceding `memory_size` rows,
excluding the query itself. Training may shuffle query rows but never their
causal memory. The fitted embedding gives calibration weight
`exp((dot(query, key) - dot(query, query)) / temperature)` and test weight 1.
The common shift cancels in the normalized conformal distribution. This is an
error-relevance surrogate, not the unpublished author attention implementation.

The score is an approximate surrogate distance, not a proven solution of the
infimum in Eq. (6). Logs include remaining inversion error separately for every
fold. Neither learned weights nor time-series dependence automatically satisfy
the assumptions needed for exact finite-sample coverage. Bounds enclose the
latent ball; that fact alone does not guarantee empirical coverage of outcomes.
Float64 bounds include an outward numerical margin but are not a formal
floating-point certification. Check coverage, inversion diagnostics, infinite
interval counts, and widths before using these results in a comparison.

To claim reproduction of the complete paper method, obtain the authors' working
attention and weighted-gradient implementation and replace the documented
adaptation. The Chronos export and evaluation contracts can be reused.
