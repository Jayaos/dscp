import torch


class QuantileRegressionRNN(torch.nn.Module):

    def __init__(self,
                 rnn_type: str,
                 dim_feature: int, 
                 dim_model: int, 
                 num_layers: int, 
                 target_quantiles: list,
                 prediction_step: int,
                 dropout: float = 0.1,
                 current_feature_dim: int=0,
                 batch_first: bool=True):
        super(QuantileRegressionRNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.dim_model = dim_model
        self.target_quantiles = target_quantiles

        rnn_cls = {"rnn": torch.nn.RNN, "gru": torch.nn.GRU, "lstm": torch.nn.LSTM}[self.rnn_type]
        self.rnn = rnn_cls(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # this will work as embedding layer for features
        self.input_linear = torch.nn.Linear(dim_feature, dim_model)
        if current_feature_dim == 0:
            self.output_linear = torch.nn.Linear(dim_model, 2*len(target_quantiles)*prediction_step) # no activation
        else:
            self.output_linear = torch.nn.Linear(dim_model+current_feature_dim, 
                                                 2*len(target_quantiles)*prediction_step) # no activation
            
    def forward(self, x, current_feature=None):

        B, T, D = x.shape
        x_emb = self.input_linear(x)
        # (batch_size, window_len, model_dim)
        h, _ = self.rnn(x_emb)
        
        if current_feature != None:
            # (batch_size, window_len, model_dim+current_feature_dim)
            h = torch.cat([h, current_feature.repeat(1, T, 1)], dim=-1)

        return self.output_linear(h)

    @staticmethod
    def get_predicted_quantile_values(model, x, current_feature=None):

        if current_feature != None:
            # (batch_size, window_size, len(target_quantiles))
            out = model(x, current_feature=current_feature)
        else:
            # (batch_size, window_size, len(target_quantiles))
            out = model(x)
            
        return out[:, -1, :] # (batch_size, 2*len(target_quantiles))