# ResCP implementation provenance

This adapter follows the executable sampling baseline in
`reservoir-conformal-prediction`, commit
`1d8e560b77890ee1fc7acad591d33b7b3e4b694f`.
Paper: https://arxiv.org/abs/2510.05060.
The upstream MIT copyright and permission notice are retained in `LICENSE`.

Reference files inside `reservoir_conformal_prediction/`:

- `src/torch_reservoir_computing/reservoir.py`: sparse initialization and
  `h_t = (1-leak) h_(t-1) + tanh(W h_(t-1) + Win r_t)`.
- `src/lib/nn/utils.py`: residual input scaling and state normalization.
- `run_RExCP_sampling.py`: pre-observation state / signed residual alignment.
- `src/reservoir_conformal_residual_sampler.py`: cosine softmax similarity,
  oldest-first linear weights `[0, ..., n-1]`, multinomial resampling, empirical
  quantiles, and minimum-width beta search.

The recurrence is explicitly named `upstream`: its tanh term has no additional
leak multiplier. The linear recency ramp follows the code. These choices must
not be silently replaced by the paper's recurrence or inverse-age weighting.

DSCP-specific corrections and choices:

- Prediction and observation are separate operations. The scaler is fitted
  once on the initial calibration prefix; the residual pool stays in raw units.
- The calibration buffer holds at most the configured number of pairs. The
  Monte Carlo sample count stays fixed (default: cap, or initial prefix length
  for unbounded memory), correcting upstream growth at the cap boundary.
- Local reservoir and sampling RNG streams make results independent of other
  series/process random draws. Sampling uses float64 raw residuals and quantiles;
  ESN states/weights use float32. Bitwise reproduction of full upstream runs is
  therefore not claimed.
- Zero states have zero cosine similarity. Singleton memory has unit sampling
  weight. A sparse zero-radius draw gets one self-loop before spectral scaling.
- Log-space weighting protects tiny temperatures. For alpha below 0.004, the
  beta-grid endpoint epsilon is reduced to alpha/4 when necessary.
- Residual clipping, ACI, learned readouts, bidirectional states, and upstream
  data loading/training are excluded from this baseline.
