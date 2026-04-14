from typing import Optional, Tuple

import torch

from .iqn import ImplicitQuantileNetwork


class IQNRNN(torch.nn.Module):
    """
    RNN encoder with an IQN head for quantile prediction.

    Input:
        src: (batch_size, window_size, dim_feature)

    Output:
        quantile_values: (batch_size, num_taus) if output_dim == 1
        taus: (batch_size, num_taus)
    """

    def __init__(
        self,
        rnn_type: str,
        dim_feature: int,
        dim_model: int,
        num_layers: int,
        current_feature_dim: int = 0,
        iqn_hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        n_cos_embedding: int = 64,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.dim_model = dim_model

        rnn_cls = {
            "rnn": torch.nn.RNN,
            "gru": torch.nn.GRU,
            "lstm": torch.nn.LSTM,
        }[self.rnn_type]
        self.rnn = rnn_cls(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.input_linear = torch.nn.Linear(dim_feature, dim_model)
        self.iqn = ImplicitQuantileNetwork(
            input_dim=dim_model + current_feature_dim,
            hidden_dim=iqn_hidden_dim,
            output_dim=output_dim,
            n_cos_embedding=n_cos_embedding,
            dropout=dropout,
        )

    def forward(
        self,
        src: torch.Tensor,
        current_feature: Optional[torch.Tensor] = None,
        taus: Optional[torch.Tensor] = None,
        num_taus: int = 1,
        return_repr: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_repr = self.encode(src, current_feature=current_feature)
        quantile_values, taus = self.iqn(
            hidden_repr=hidden_repr,
            taus=taus,
            num_taus=num_taus,
        )

        if return_repr:
            return quantile_values, taus, hidden_repr
        return quantile_values, taus

    def encode(
        self,
        src: torch.Tensor,
        current_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_emb = self.input_linear(src)
        hidden_states, _ = self.rnn(src_emb)
        if current_feature is not None:
            hidden_states = torch.cat(
                [hidden_states, current_feature.repeat(1, hidden_states.shape[1], 1)],
                dim=-1,
            )
        return hidden_states[:, -1, :]

    @torch.no_grad()
    def predict_quantiles(
        self,
        src: torch.Tensor,
        quantiles: torch.Tensor,
        current_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_repr = self.encode(src, current_feature=current_feature)
        return self.iqn.predict_quantiles(hidden_repr, quantiles)

    @staticmethod
    def get_predicted_quantile_values(
        model,
        x: torch.Tensor,
        quantiles: torch.Tensor,
        current_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return model.predict_quantiles(
            src=x,
            quantiles=quantiles,
            current_feature=current_feature,
        )
