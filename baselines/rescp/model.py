"""Causal reservoir conformal residual intervals.

Adapted from Reservoir Conformal Prediction, upstream commit
1d8e560b77890ee1fc7acad591d33b7b3e4b694f (see LICENSE and UPSTREAM.md).
The recurrence, cosine similarity, recency ramp, sampling, and beta grid
follow the executable upstream sampling baseline. The memory cap and random
streams are corrected for streaming use in DSCP.
"""

from numbers import Integral, Real

import numpy as np
from scipy import sparse
import torch


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _number(name, value, minimum=None, maximum=None, strict_min=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and (value <= minimum if strict_min else value < minimum):
        comparison = "greater than" if strict_min else "at least"
        raise ValueError(f"{name} must be {comparison} {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


class ResCPResidualIntervalEstimator:
    """Estimate signed residual bounds, observing each outcome after prediction.

    ``fit`` freezes a scaler on the supplied calibration prefix. Each stored
    pair contains the state *before* its residual was observed. The recurrent
    state remains unnormalized; only the copy used for similarity is normalized.
    ``predict_interval`` consumes sampling randomness but never observes a label
    or changes the reservoir/memory. ``observe`` advances these after prediction.
    """

    def __init__(
        self,
        reservoir_size=512,
        spectral_radius=1.2,
        leak_rate=0.9,
        input_scaling=0.25,
        connectivity=0.2,
        temperature=0.1,
        calibration_size=3800,
        sampling_num=None,
        use_beta_search=True,
        beta_bins=100,
        decay="linear",
        seed=2026,
        sampling_seed=None,
        decay_rate=0.99,
        recurrence="upstream",
    ):
        self.reservoir_size = _integer("reservoir_size", reservoir_size)
        self.spectral_radius = _number("spectral_radius", spectral_radius, minimum=0)
        self.leak_rate = _number("leak_rate", leak_rate, minimum=0, maximum=1, strict_min=True)
        self.input_scaling = _number("input_scaling", input_scaling, minimum=0)
        self.connectivity = _number("connectivity", connectivity, minimum=0, maximum=1, strict_min=True)
        self.temperature = _number("temperature", temperature, minimum=0, strict_min=True)
        self.calibration_size = None if calibration_size is None else _integer("calibration_size", calibration_size)
        self.sampling_num = None if sampling_num is None else _integer("sampling_num", sampling_num)
        if not isinstance(use_beta_search, (bool, np.bool_)):
            raise ValueError("use_beta_search must be a boolean.")
        self.use_beta_search = bool(use_beta_search)
        self.beta_bins = _integer("beta_bins", beta_bins)
        if decay not in ("linear", "none", "exponential"):
            raise ValueError("decay must be 'linear', 'none', or 'exponential'.")
        self.decay = decay
        self.decay_rate = _number("decay_rate", decay_rate, minimum=0, maximum=1, strict_min=True)
        if recurrence != "upstream":
            raise ValueError("Only recurrence='upstream' is supported.")
        self.recurrence = recurrence
        self.seed = self._validate_seed("seed", seed)
        self.sampling_seed = self._validate_seed(
            "sampling_seed", (self.seed + 1) % (2 ** 32) if sampling_seed is None else sampling_seed
        )
        self._fitted = False

    @staticmethod
    def _validate_seed(name, value):
        value = _integer(name, value, minimum=0)
        if value >= 2 ** 32:
            raise ValueError(f"{name} must be less than 2**32.")
        return value

    def _initialize_reservoir(self):
        # Local RandomState preserves SciPy's upstream sparse.rand initialization
        # without resetting NumPy or Torch's process-wide random generators.
        rng = np.random.RandomState(self.seed)
        weights = sparse.rand(
            self.reservoir_size, self.reservoir_size,
            density=self.connectivity, random_state=rng,
        ).toarray()
        weights[weights > 0] -= 0.5
        if self.spectral_radius == 0:
            weights.fill(0)
        else:
            radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
            if radius <= np.finfo(float).eps:
                # Very small sparse reservoirs may be zero/nilpotent. A self-loop
                # gives a defined radius instead of upstream division by zero.
                weights[0, 0] = 1.0
                radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
            weights *= self.spectral_radius / radius
        inputs = (2.0 * rng.binomial(1, 0.5, self.reservoir_size) - 1.0) * self.input_scaling
        self._internal_weights = torch.tensor(weights, dtype=torch.float32)
        self._input_weights = torch.tensor(inputs, dtype=torch.float32)
        if not torch.isfinite(self._internal_weights).all() or not torch.isfinite(self._input_weights).all():
            raise ValueError("Reservoir weights cannot be represented as finite float32 values.")
        self._state = torch.zeros(self.reservoir_size, dtype=torch.float32)
        self._sampling_generator = torch.Generator(device="cpu")
        self._sampling_generator.manual_seed(self.sampling_seed)

    def fit(self, calibration_residuals, normalize=True):
        """Initialize from a nonempty chronological prefix, then return ``self``."""
        if not isinstance(normalize, (bool, np.bool_)):
            raise ValueError("normalize must be a boolean.")
        residuals = np.asarray(calibration_residuals)
        if residuals.ndim == 2 and residuals.shape[1] == 1:
            residuals = residuals[:, 0]
        if residuals.ndim != 1 or residuals.size == 0 or residuals.dtype.kind not in "iuf":
            raise ValueError("calibration_residuals must be a nonempty numeric residual vector.")
        residuals = residuals.astype(np.float64)
        if not np.isfinite(residuals).all():
            raise ValueError("calibration_residuals must contain only finite values.")
        mean = float(residuals.mean()) if normalize else 0.0
        std = float(residuals.std()) if normalize else 1.0
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError("Calibration normalization statistics must be finite.")
        self._input_mean = mean
        self._input_std = std if std > 0 else 1.0
        self._effective_sampling_num = (
            self.sampling_num if self.sampling_num is not None
            else self.calibration_size if self.calibration_size is not None
            else len(residuals)
        )
        self._fitted = False
        self._initialize_reservoir()
        capacity = self.calibration_size if self.calibration_size is not None else len(residuals)
        self._memory_states = torch.empty((capacity, self.reservoir_size), dtype=torch.float32)
        self._memory_residuals = torch.empty(capacity, dtype=torch.float64)
        self._memory_start = 0
        self._memory_size = 0
        self._fitted = True
        try:
            for residual in residuals:
                self.observe(float(residual))
        except Exception:
            self._fitted = False
            raise
        return self

    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError("Fit the ResCP estimator before using it.")

    @property
    def input_mean(self):
        self._require_fit()
        return self._input_mean

    @property
    def input_std(self):
        self._require_fit()
        return self._input_std

    @property
    def memory_size(self):
        return self._memory_size if self._fitted else 0

    @property
    def effective_sampling_num(self):
        self._require_fit()
        return self._effective_sampling_num

    @staticmethod
    def _normalized(state):
        norm = torch.linalg.vector_norm(state)
        return state / norm if norm > 0 else torch.zeros_like(state)

    def _ordered_memory(self):
        """Return chronological states and raw residuals, oldest first."""
        self._require_fit()
        end = self._memory_start + self._memory_size
        if end <= len(self._memory_residuals):
            return self._memory_states[self._memory_start:end], self._memory_residuals[self._memory_start:end]
        split = end % len(self._memory_residuals)
        return (
            torch.cat((self._memory_states[self._memory_start:], self._memory_states[:split])),
            torch.cat((self._memory_residuals[self._memory_start:], self._memory_residuals[:split])),
        )

    def _append(self, state, residual):
        capacity = len(self._memory_residuals)
        if self._memory_size == capacity and self.calibration_size is None:
            self._memory_states = torch.cat((self._memory_states, torch.empty_like(self._memory_states)))
            self._memory_residuals = torch.cat((self._memory_residuals, torch.empty_like(self._memory_residuals)))
            capacity *= 2
        index = (self._memory_start + self._memory_size) % capacity
        self._memory_states[index] = state
        self._memory_residuals[index] = residual
        if self._memory_size == capacity:
            self._memory_start = (self._memory_start + 1) % capacity
        else:
            self._memory_size += 1

    def observe(self, residual):
        """Store the forecast-time state with the revealed residual, then advance."""
        self._require_fit()
        residual = _number("residual", residual)
        encoded = (residual - self._input_mean) / self._input_std
        if not np.isfinite(encoded) or abs(encoded) > np.finfo(np.float32).max:
            raise ValueError("Standardized residual cannot be represented as a finite float32 value.")
        next_state = (1.0 - self.leak_rate) * self._state + torch.tanh(
            self._internal_weights @ self._state + self._input_weights * encoded
        )
        if not torch.isfinite(next_state).all():
            raise ValueError("Reservoir update produced a nonfinite state.")
        self._append(self._normalized(self._state), residual)
        self._state = next_state
        return self

    def _sampling_weights(self):
        self._require_fit()
        query = self._normalized(self._state)
        count = self._memory_size
        first_count = min(count, len(self._memory_residuals) - self._memory_start)
        similarity = self._memory_states[
            self._memory_start:self._memory_start + first_count
        ] @ query
        if first_count < count:
            # Multiply the ring's two views separately: copying an entire W x H
            # state matrix on every forecast would dominate memory traffic.
            similarity = torch.cat((
                similarity, self._memory_states[:count - first_count] @ query,
            ))
        similarity = similarity.to(torch.float64)
        # Add recency in log space to preserve the upstream product while
        # avoiding zero total mass at very small temperatures.
        log_decay = torch.zeros(count, dtype=torch.float64)
        if self.decay == "linear" and count > 1:
            log_decay = torch.log(torch.arange(count, dtype=torch.float64))
        elif self.decay == "exponential":
            age = torch.arange(count, 0, -1, dtype=torch.float64)
            log_decay = age * np.log(self.decay_rate)
        eligible = torch.isfinite(log_decay)
        similarity = similarity - torch.max(similarity[eligible])
        logits = similarity / self.temperature + log_decay
        logits[~eligible] = -torch.inf
        return torch.softmax(logits, dim=0)

    def predict_interval(self, confidence_pair):
        """Return ``(lower_residual, upper_residual, beta)`` before observing y."""
        self._require_fit()
        pair = np.asarray(confidence_pair)
        if pair.shape != (2,) or pair.dtype.kind not in "iuf":
            raise ValueError("confidence_pair must contain two quantile probabilities.")
        lower, upper = map(float, pair)
        if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower < upper <= 1):
            raise ValueError("confidence_pair must satisfy 0 <= lower < upper <= 1.")
        coverage = upper - lower
        alpha = 1.0 - coverage
        if alpha <= 0:
            raise ValueError("confidence_pair must leave positive miscoverage probability.")
        indices = torch.multinomial(
            self._sampling_weights(), self.effective_sampling_num,
            replacement=True, generator=self._sampling_generator,
        )
        # Sampling weights are chronological; map their indices into the ring
        # and gather only the residual sample, leaving the states in place.
        physical_indices = (indices + self._memory_start) % len(self._memory_residuals)
        sampled = self._memory_residuals[physical_indices]
        if self.use_beta_search and self.beta_bins > 1:
            epsilon = min(1.0e-3, alpha / 4)
            betas = torch.linspace(epsilon, alpha - epsilon, self.beta_bins, dtype=torch.float64)
            upper_betas = torch.clamp(coverage + betas, max=1.0)
        elif self.use_beta_search:
            betas = torch.tensor([alpha / 2], dtype=torch.float64)
            upper_betas = coverage + betas
        else:
            betas = torch.tensor([lower], dtype=torch.float64)
            upper_betas = torch.tensor([upper], dtype=torch.float64)
        quantiles = torch.quantile(sampled, torch.cat((betas, upper_betas)), interpolation="linear")
        low, high = quantiles.chunk(2)
        selected = int(torch.argmin(high - low))
        return float(low[selected]), float(high[selected]), float(betas[selected])
