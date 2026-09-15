from math import expm1, log, pi
from numbers import Integral
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


COSINE_EMBEDDING_HEAD = "cosine_embedding"
PARTIALLY_MONOTONIC_HEAD = "partially_monotonic"
SUPPORTED_PREDICTION_HEADS = (
    COSINE_EMBEDDING_HEAD,
    PARTIALLY_MONOTONIC_HEAD,
)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class QuantileEmbedding(torch.nn.Module):
    """
    Cosine quantile embedding used by implicit quantile networks.
    """

    def __init__(self, output_dim: int, n_cos_embedding: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.n_cos_embedding = n_cos_embedding
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(n_cos_embedding, n_cos_embedding),
            torch.nn.PReLU(),
            torch.nn.Linear(n_cos_embedding, output_dim),
        )

    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        cos_embedded_tau = self.cos_embed(taus)
        return self.output_layer(cos_embedded_tau)

    def cos_embed(self, taus: torch.Tensor) -> torch.Tensor:
        integers = torch.arange(
            self.n_cos_embedding,
            device=taus.device,
            dtype=taus.dtype,
        )
        return torch.cos(pi * taus.unsqueeze(-1) * integers)


class ImplicitQuantileNetwork(torch.nn.Module):
    """
    Legacy cosine-embedding IQN head for RNN or Transformer representations.

    Its prediction method retains the existing sampling-based rearrangement
    behavior for backward compatibility.

    Input:
        hidden_repr: (B, D) or (B, T, D)
        taus: (B, N) or (N,)

    Output:
        quantile_values: (B, N)
        taus: (B, N)
    """

    head_type = COSINE_EMBEDDING_HEAD

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        n_cos_embedding: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim

        if self.hidden_dim == input_dim:
            self.input_projection = torch.nn.Identity()
        else:
            self.input_projection = torch.nn.Linear(input_dim, self.hidden_dim)

        self.quantile_embedding = QuantileEmbedding(
            output_dim=self.hidden_dim,
            n_cos_embedding=n_cos_embedding,
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        hidden_repr: torch.Tensor,
        taus: Optional[torch.Tensor] = None,
        num_taus: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_repr = self._prepare_hidden_repr(hidden_repr)
        hidden_repr = self.input_projection(hidden_repr)

        if taus is None:
            taus = self.sample_taus(
                batch_size=hidden_repr.shape[0],
                num_taus=num_taus,
                device=hidden_repr.device,
                dtype=hidden_repr.dtype,
            )
        else:
            taus = self._prepare_taus(
                taus=taus,
                batch_size=hidden_repr.shape[0],
                device=hidden_repr.device,
                dtype=hidden_repr.dtype,
            )

        embedded_taus = self.quantile_embedding(taus)
        conditioned_hidden = hidden_repr.unsqueeze(1) * (1.0 + embedded_taus)
        quantile_values = self.output_layer(conditioned_hidden).squeeze(-1)

        return quantile_values, taus

    @torch.no_grad()
    def predict_quantiles(
        self,
        hidden_repr: torch.Tensor,
        quantiles: torch.Tensor,
        sampling_num: int = 1000,
    ) -> torch.Tensor:
        batch_size = hidden_repr.shape[0]
        quantiles = self._prepare_taus(
            taus=quantiles,
            batch_size=batch_size,
            device=hidden_repr.device,
            dtype=hidden_repr.dtype,
        )

        if sampling_num < 1:
            raise ValueError("sampling_num must be a positive integer.")

        sampled_values, _ = self(hidden_repr, taus=None, num_taus=sampling_num)
        return torch.stack(
            [
                torch.quantile(sampled_values[i], q=quantiles[i], dim=0)
                for i in range(batch_size)
            ],
            dim=0,
        )

    @staticmethod
    def sample_taus(
        batch_size: int,
        num_taus: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        num_taus = _positive_integer(num_taus, "num_taus")
        return torch.rand(batch_size, num_taus, device=device, dtype=dtype)

    @staticmethod
    def _prepare_hidden_repr(hidden_repr: torch.Tensor) -> torch.Tensor:
        if hidden_repr.ndim == 3:
            return hidden_repr[:, -1, :]
        if hidden_repr.ndim == 2:
            return hidden_repr
        raise ValueError(
            "hidden_repr must have shape (batch_size, hidden_dim) "
            "or (batch_size, seq_len, hidden_dim)."
        )

    @staticmethod
    def _prepare_taus(
        taus: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(taus):
            taus = torch.tensor(taus, device=device, dtype=dtype)
        else:
            taus = taus.to(device=device, dtype=dtype)

        if taus.ndim == 0:
            taus = taus.view(1, 1).expand(batch_size, 1)
        elif taus.ndim == 1:
            taus = taus.unsqueeze(0).expand(batch_size, -1)
        elif taus.ndim == 2:
            if taus.shape[0] == 1 and batch_size > 1:
                taus = taus.expand(batch_size, -1)
            elif taus.shape[0] != batch_size:
                raise ValueError(
                    "When taus is 2D, its first dimension must match batch size."
                )
        else:
            raise ValueError("taus must have shape (), (num_taus,) or (batch_size, num_taus).")

        if torch.any(~torch.isfinite(taus)):
            raise ValueError("taus must be finite.")
        if torch.any((taus < 0.0) | (taus > 1.0)):
            raise ValueError("taus must lie in [0, 1].")

        return taus


def _inverse_softplus(value: float) -> float:
    """Return x such that softplus(x) == value for a positive value."""
    return log(expm1(value))


class NonnegativeLinear(torch.nn.Module):
    """Linear map without bias whose effective weights are strictly positive."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = _positive_integer(in_features, "in_features")
        self.out_features = _positive_integer(out_features, "out_features")
        self.raw_weight = torch.nn.Parameter(torch.empty(out_features, in_features))

        # Initialize each row with an effective weight sum close to one.
        target_weight = 1.0 / in_features
        torch.nn.init.normal_(
            self.raw_weight,
            mean=_inverse_softplus(target_weight),
            std=0.02,
        )

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, bias=None)


class PartiallyMonotonicQuantileHead(torch.nn.Module):
    """Quantile head that is nondecreasing in tau for every fixed context.

    The implementation follows

        r_1(tau) = phi_1(A_1 h + b_1^+ tau + c_1)
        r_k(tau) = phi_k(A_k h + B_k^+ r_{k-1}(tau) + c_k)
        g(h, tau) = a(h) + (v^+)^T r_K(tau),

    where the tau-path parameters b_1^+, B_k^+, and v^+ are represented by
    softplus transforms and are therefore elementwise nonnegative. Context
    projections and biases remain unrestricted.
    """

    head_type = PARTIALLY_MONOTONIC_HEAD

    _ACTIVATIONS = {
        "sigmoid": torch.nn.Sigmoid,
        "softplus": torch.nn.Softplus,
        "tanh": torch.nn.Tanh,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "tanh",
    ):
        super().__init__()

        self.input_dim = _positive_integer(input_dim, "input_dim")
        if isinstance(hidden_dims, (str, bytes)):
            raise TypeError("hidden_dims must be a sequence of positive integers.")

        self.hidden_dims = tuple(
            _positive_integer(dim, "each hidden dimension")
            for dim in hidden_dims
        )
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one positive integer.")

        activation_name = str(activation).strip().lower()
        if activation_name not in self._ACTIVATIONS:
            supported = ", ".join(sorted(self._ACTIVATIONS))
            raise ValueError(
                f"Unsupported monotonic activation {activation!r}. "
                f"Choose one of: {supported}."
            )
        self.activation_name = activation_name
        self.activations = torch.nn.ModuleList(
            self._ACTIVATIONS[activation_name]() for _ in self.hidden_dims
        )

        # A_k h + c_k; both A_k and c_k are intentionally unrestricted.
        self.context_layers = torch.nn.ModuleList(
            torch.nn.Linear(self.input_dim, hidden_dim)
            for hidden_dim in self.hidden_dims
        )

        # b_1^+ tau.
        self.raw_tau_weight = torch.nn.Parameter(torch.empty(self.hidden_dims[0]))
        torch.nn.init.normal_(
            self.raw_tau_weight,
            mean=_inverse_softplus(1.0),
            std=0.02,
        )

        # B_k^+ r_{k-1}(tau), for k = 2, ..., K.
        self.positive_hidden_layers = torch.nn.ModuleList(
            NonnegativeLinear(previous_dim, hidden_dim)
            for previous_dim, hidden_dim in zip(
                self.hidden_dims[:-1],
                self.hidden_dims[1:],
            )
        )

        # a(h) = w_a^T h + b_a and the nonnegative v^+ output weights.
        self.base_layer = torch.nn.Linear(self.input_dim, 1)
        self.raw_output_weight = torch.nn.Parameter(
            torch.empty(self.hidden_dims[-1])
        )
        torch.nn.init.normal_(
            self.raw_output_weight,
            mean=_inverse_softplus(1.0 / self.hidden_dims[-1]),
            std=0.02,
        )

    @property
    def tau_weight(self) -> torch.Tensor:
        return F.softplus(self.raw_tau_weight)

    @property
    def output_weight(self) -> torch.Tensor:
        return F.softplus(self.raw_output_weight)

    def positive_raw_parameters(self):
        """Yield raw parameters whose softplus transforms form positive paths."""
        yield self.raw_tau_weight
        for layer in self.positive_hidden_layers:
            yield layer.raw_weight
        yield self.raw_output_weight

    def forward(
        self,
        hidden_repr: torch.Tensor,
        taus: Optional[torch.Tensor] = None,
        num_taus: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_repr = ImplicitQuantileNetwork._prepare_hidden_repr(hidden_repr)

        if taus is None:
            taus = self.sample_taus(
                batch_size=hidden_repr.shape[0],
                num_taus=num_taus,
                device=hidden_repr.device,
                dtype=hidden_repr.dtype,
            )
        else:
            taus = ImplicitQuantileNetwork._prepare_taus(
                taus=taus,
                batch_size=hidden_repr.shape[0],
                device=hidden_repr.device,
                dtype=hidden_repr.dtype,
            )
            if torch.any((taus <= 0.0) | (taus >= 1.0)):
                raise ValueError(
                    "The partially monotonic head requires taus to lie in (0, 1)."
                )

        context_term = self.context_layers[0](hidden_repr).unsqueeze(1)
        tau_term = taus.unsqueeze(-1) * self.tau_weight
        representation = self.activations[0](context_term + tau_term)

        for layer_index, positive_layer in enumerate(
            self.positive_hidden_layers,
            start=1,
        ):
            context_term = self.context_layers[layer_index](hidden_repr).unsqueeze(1)
            representation = self.activations[layer_index](
                context_term + positive_layer(representation)
            )

        base_value = self.base_layer(hidden_repr)
        monotonic_value = torch.sum(
            representation * self.output_weight,
            dim=-1,
        )
        quantile_values = base_value + monotonic_value
        return quantile_values, taus

    @torch.no_grad()
    def predict_quantiles(
        self,
        hidden_repr: torch.Tensor,
        quantiles: torch.Tensor,
        sampling_num: int = 1000,
    ) -> torch.Tensor:
        # sampling_num is accepted for a shared API with the cosine IQN head.
        # This head evaluates g(h, tau) directly and performs no MC rearrangement.
        del sampling_num
        quantile_values, _ = self(hidden_repr, taus=quantiles)
        return quantile_values

    @staticmethod
    def sample_taus(
        batch_size: int,
        num_taus: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        num_taus = _positive_integer(num_taus, "num_taus")
        taus = torch.rand(batch_size, num_taus, device=device, dtype=dtype)
        # torch.rand is already strictly below one. Replace only a possible
        # exact zero so samples remain in the open unit interval without
        # unnecessarily trimming the upper tail for lower-precision dtypes.
        return taus.clamp_min(torch.finfo(dtype).tiny)


def build_quantile_head(
    prediction_head: str,
    input_dim: int,
    hidden_dim: Optional[int] = None,
    n_cos_embedding: int = 64,
    dropout: float = 0.1,
    monotonic_num_layers: int = 1,
    monotonic_hidden_dims: Optional[Sequence[int]] = None,
    monotonic_activation: str = "tanh",
) -> torch.nn.Module:
    """Build one of the supported conditional-quantile prediction heads."""
    head_name = str(prediction_head).strip().lower()

    if head_name == COSINE_EMBEDDING_HEAD:
        return ImplicitQuantileNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_cos_embedding=n_cos_embedding,
            dropout=dropout,
        )

    if head_name == PARTIALLY_MONOTONIC_HEAD:
        if monotonic_hidden_dims is None:
            monotonic_num_layers = _positive_integer(
                monotonic_num_layers,
                "monotonic_num_layers",
            )
            monotonic_width = input_dim if hidden_dim is None else hidden_dim
            monotonic_hidden_dims = [monotonic_width] * monotonic_num_layers
        return PartiallyMonotonicQuantileHead(
            input_dim=input_dim,
            hidden_dims=monotonic_hidden_dims,
            activation=monotonic_activation,
        )

    supported = ", ".join(SUPPORTED_PREDICTION_HEADS)
    raise ValueError(
        f"Unsupported prediction_head {prediction_head!r}. Choose one of: {supported}."
    )


def build_iqn_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Build AdamW while avoiding inverse decay of effective positive weights.

    AdamW decay toward zero is appropriate for ordinary parameters. For a raw
    parameter p used as softplus(p), however, decay toward raw zero moves its
    effective value toward softplus(0), rather than toward effective zero. The
    raw parameters that enforce monotonicity therefore use zero weight decay.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative.")

    monotonic_heads = [
        module
        for module in model.modules()
        if isinstance(module, PartiallyMonotonicQuantileHead)
    ]
    if not monotonic_heads:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    positive_raw_parameters = []
    seen_positive_parameter_ids = set()
    for quantile_head in monotonic_heads:
        for parameter in quantile_head.positive_raw_parameters():
            if id(parameter) not in seen_positive_parameter_ids:
                positive_raw_parameters.append(parameter)
                seen_positive_parameter_ids.add(id(parameter))
    positive_parameter_ids = {
        id(parameter) for parameter in positive_raw_parameters
    }
    ordinary_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in positive_parameter_ids
    ]

    return torch.optim.AdamW(
        [
            {
                "params": ordinary_parameters,
                "weight_decay": weight_decay,
            },
            {
                "params": positive_raw_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
    )
