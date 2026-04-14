import torch
import math
import torch.nn.functional as F
from utils.utils import get_sorted_unique_quantiles


class QuantileRegressionTransformer(torch.nn.Module):
    """
    stacked Transformer for quantile prediction
    Input: sequential features for endoder padded by max_seq_len and sequential features decoder
        sequential features: (batch_size * past_window * feature_dim)
    Output: predicted values for pre-defined quantiles (batch_size * step_prediction * # of pre-defined quantiles)
    """
    
    def __init__(self, 
                 dim_feature: int, 
                 dim_model: int, 
                 num_head: int, 
                 dim_ff: int, 
                 num_layers: int, 
                 target_quantiles: list,
                 prediction_step: int,
                 dropout: float = 0.1,
                 current_feature_dim: int = 0,
                 batch_first: bool=True):
        super(QuantileRegressionTransformer, self).__init__()
        self.dim_model = dim_model
        self.target_quantiles = target_quantiles
        self.sorted_quantiles = get_sorted_unique_quantiles(target_quantiles)
        self.num_quantiles = len(self.sorted_quantiles)
        self.prediction_step = prediction_step
        self.positional_encoding = PositionalEncoding(dim_model, dropout, batch_first=batch_first)

        encoder_Layer = torch.nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_head, dim_feedforward=dim_ff, 
                                                   dropout=dropout, batch_first=batch_first)
        self.encoder = torch.nn.TransformerEncoder(encoder_Layer, num_layers=num_layers)

        # this will work as embedding layer for features
        self.input_linear = torch.nn.Linear(dim_feature, dim_model)
        if current_feature_dim == 0:
            head_input_dim = dim_model
        else:
            head_input_dim = dim_model + current_feature_dim

        self.base_head = torch.nn.Linear(head_input_dim, prediction_step)
        self.increment_head = torch.nn.Linear(
            head_input_dim, prediction_step * max(self.num_quantiles - 1, 0)
        )

    def forward(self, src, src_mask, src_key_padding_mask, current_feature=None):

        B, T, D = src.shape
        src_emb = self.input_linear(src)
        src_emb = src_emb * math.sqrt(self.dim_model)
        src_emb = self.positional_encoding(src_emb)
        # (batch_size, window_len, model_dim)
        h = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        if current_feature != None:
            # (batch_size, window_len, model_dim+current_feature_dim)
            h = torch.cat([h, current_feature.repeat(1, T, 1)], dim=-1)

        base = self.base_head(h).unsqueeze(-1)

        if self.num_quantiles == 1:
            quantiles = base
        else:
            increments = F.softplus(self.increment_head(h))
            increments = increments.view(B, T, self.prediction_step, self.num_quantiles - 1)
            quantiles = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)

        return quantiles.reshape(B, T, self.prediction_step * self.num_quantiles)
    
    @staticmethod
    def get_predicted_quantile_values(model, x, current_feature=None):

        device = x.device
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)

        if current_feature != None:
            # (batch_size, window_size, len(target_quantiles))
            out = model(x, src_mask=causal_mask, src_key_padding_mask=None, current_feature=current_feature)
        else:
            # (batch_size, window_size, len(target_quantiles))
            out = model(x, src_mask=causal_mask, src_key_padding_mask=None) 
            
        return out[:, -1, :] # (batch_size, prediction_step * num_quantiles)
        

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
    
