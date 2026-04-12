from math import pi
from typing import Optional, Tuple

import torch


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
    IQN head that consumes hidden representations from an encoder
    such as an RNN or Transformer and outputs quantile values.

    Input:
        hidden_repr: (B, D) or (B, T, D)
        taus: (B, N) or (N,)

    Output:
        quantile_values: (B, N) when output_dim == 1, else (B, N, output_dim)
        taus: (B, N)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        n_cos_embedding: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.output_dim = output_dim

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
            torch.nn.Linear(self.hidden_dim, output_dim),
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
        quantile_values = self.output_layer(conditioned_hidden)

        if self.output_dim == 1:
            quantile_values = quantile_values.squeeze(-1)

        return quantile_values, taus

    @torch.no_grad()
    def predict_quantiles(
        self,
        hidden_repr: torch.Tensor,
        quantiles: torch.Tensor,
    ) -> torch.Tensor:
        quantile_values, _ = self(hidden_repr, taus=quantiles)
        return quantile_values

    @staticmethod
    def sample_taus(
        batch_size: int,
        num_taus: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
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

        if torch.any((taus < 0.0) | (taus > 1.0)):
            raise ValueError("taus must lie in [0, 1].")

        return taus
