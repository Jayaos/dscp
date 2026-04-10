import torch
import math
from utils.utils import generate_strided_feature


class TransformerPredictor(torch.nn.Module):
    """
    stacked Transformer for prediction
    """
    
    def __init__(self, 
                 dim_feature: int, 
                 dim_model: int, 
                 num_head: int, 
                 dim_ff: int, 
                 num_layer: int, 
                 prediction_step: int,
                 dropout: float = 0.1, 
                 batch_first: bool=True):
        super(TransformerPredictor, self).__init__()
        self.dim_model = dim_model
        self.positional_encoding = PositionalEncoding(dim_model, dropout, batch_first=batch_first)

        encoder_Layer = torch.nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_head, dim_feedforward=dim_ff, 
                                                   dropout=dropout, batch_first=batch_first)
        self.encoder = torch.nn.TransformerEncoder(encoder_Layer, num_layers=num_layer)

        # this will work as embedding layer for features
        self.input_linear = torch.nn.Linear(dim_feature, dim_model)
        self.output_linear = torch.nn.Linear(dim_model, prediction_step) # no activation

    def forward(self, src, src_mask, src_key_padding_mask, return_repr=False):

        src_emb = self.input_linear(src)
        src_emb = src_emb * math.sqrt(self.dim_model)
        src_emb = self.positional_encoding(src_emb)
        outputs = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if return_repr:
            return self.output_linear(outputs), outputs
        else:
            return self.output_linear(outputs)
    
    @staticmethod
    def encode(model, x):

        device = x.device
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)
        x_emb = model.input_linear(x)
        x_emb = x_emb * math.sqrt(model.dim_model)
        x_emb = model.positional_encoding(x_emb)

        return model.encoder(x_emb, mask=causal_mask, src_key_padding_mask=None) # (B, T, D)
    
    @staticmethod
    def encode_dataloader(model, dataloader, strided_features, device):

        repr_list = []
        residual_list = []
        for strided_x, strided_residual, strided_y, \
            target_x, target_residual, target_y, target_predictions in dataloader:

            strided_feature = generate_strided_feature(strided_x, 
                                                        strided_residual, 
                                                        strided_y,
                                                        strided_features)

            strided_feature = strided_feature.to(device)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(strided_feature.shape[1]).to(device)

            x_emb = model.input_linear(strided_feature)
            x_emb = x_emb * math.sqrt(model.dim_model)
            x_emb = model.positional_encoding(x_emb)
            repr = model.encoder(x_emb, mask=causal_mask, src_key_padding_mask=None) # (B, T, D)

            repr_list.append(repr[:,-1,:])
            residual_list.append(strided_residual[:,-1])

        return torch.vstack(repr_list), torch.vstack(residual_list)


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
    