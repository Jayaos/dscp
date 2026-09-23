import math
from typing import Optional, Sequence, Tuple

import torch

from .iqn import build_quantile_head


class IQNTransformer(torch.nn.Module):
    """
    Transformer encoder with a selectable conditional-quantile prediction head.

    Input:
        src: (batch_size, window_size, dim_feature)

    Output:
        quantile_values: (batch_size, num_taus)
        taus: (batch_size, num_taus)
    """

    def __init__(
        self,
        dim_feature: int,
        dim_model: int,
        num_head: int,
        dim_ff: int,
        num_layers: int,
        current_feature_dim: int = 0,
        iqn_hidden_dim: Optional[int] = None,
        n_cos_embedding: int = 64,
        dropout: float = 0.1,
        batch_first: bool = True,
        prediction_head: str = "cosine_embedding",
        monotonic_num_layers: int = 1,
        monotonic_hidden_dims: Optional[Sequence[int]] = None,
        monotonic_activation: str = "tanh",
        interval_mode: str = "sampling",
        sampling_num: int = 1000,
        iqn_num_layers: int = 1,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.positional_encoding = PositionalEncoding(
            dim_model,
            dropout,
            batch_first=batch_first,
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.input_linear = torch.nn.Linear(dim_feature, dim_model)
        self.iqn = build_quantile_head(
            prediction_head=prediction_head,
            input_dim=dim_model + current_feature_dim,
            hidden_dim=iqn_hidden_dim,
            n_cos_embedding=n_cos_embedding,
            dropout=dropout,
            monotonic_num_layers=monotonic_num_layers,
            monotonic_hidden_dims=monotonic_hidden_dims,
            monotonic_activation=monotonic_activation,
            interval_mode=interval_mode,
            sampling_num=sampling_num,
            iqn_num_layers=iqn_num_layers,
        )
        self.prediction_head = self.iqn.head_type
        self.interval_mode = self.iqn.interval_mode
        self.sampling_num = getattr(self.iqn, "sampling_num", None)
        self.iqn_num_layers = getattr(self.iqn, "iqn_num_layers", None)

    def forward(
        self,
        src: torch.Tensor,
        current_feature: Optional[torch.Tensor] = None,
        taus: Optional[torch.Tensor] = None,
        num_taus: int = 1,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_repr: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_repr = self.encode(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            current_feature=current_feature,
        )
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
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        current_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_emb = self.input_linear(src)
        src_emb = src_emb * math.sqrt(self.dim_model)
        src_emb = self.positional_encoding(src_emb)
        hidden_states = self.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        if current_feature is not None:
            hidden_states = torch.cat(
                [hidden_states, current_feature.repeat(1, hidden_states.shape[1], 1)],
                dim=-1,
            )

        return hidden_states[:, -1, :] # output the representation of the last timestep

    @torch.no_grad()
    def predict_quantiles(
        self,
        src: torch.Tensor,
        quantiles: torch.Tensor,
        current_feature: Optional[torch.Tensor] = None,
        sampling_num: Optional[int] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            src.shape[1],
            device=src.device,
        )
        hidden_repr = self.encode(
            src=src,
            src_mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            current_feature=current_feature,
        )
        return self.iqn.predict_quantiles(
            hidden_repr,
            quantiles,
            sampling_num=sampling_num,
        )

    @staticmethod
    def get_predicted_quantile_values(
        model,
        x: torch.Tensor,
        quantiles: torch.Tensor,
        current_feature: Optional[torch.Tensor] = None,
        sampling_num: Optional[int] = None,
    ) -> torch.Tensor:
        return model.predict_quantiles(
            src=x,
            quantiles=quantiles,
            current_feature=current_feature,
            sampling_num=sampling_num,
        )


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:x.size(1)].transpose(0, 1)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)
