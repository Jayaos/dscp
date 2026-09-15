import torch
import torch.nn.functional as F
from utils.utils import get_sorted_unique_quantiles


class QuantileRegressionRNN(torch.nn.Module):
    """Shared recurrent encoder with selectable quantile prediction heads.

    head_type="nondecreasing" uses cumulative nonnegative increments (default).
    head_type="independent" directly predicts each quantile and allows crossing.
    """

    def __init__(self,
                 rnn_type: str,
                 dim_feature: int, 
                 dim_model: int, 
                 num_layers: int, 
                 target_quantiles: list,
                 prediction_step: int,
                 dropout: float = 0.1,
                 current_feature_dim: int=0,
                 batch_first: bool=True,
                 head_type: str = "nondecreasing"):
        super(QuantileRegressionRNN, self).__init__()
        if head_type not in ("nondecreasing", "independent"):
            raise ValueError(
                f"head_type must be 'nondecreasing' or 'independent', got {head_type!r}"
            )
        self.head_type = head_type
        self.rnn_type = rnn_type.lower()
        self.dim_model = dim_model
        self.target_quantiles = target_quantiles
        self.sorted_quantiles = get_sorted_unique_quantiles(target_quantiles)
        self.num_quantiles = len(self.sorted_quantiles)
        self.prediction_step = prediction_step

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
            head_input_dim = dim_model
        else:
            head_input_dim = dim_model + current_feature_dim

        if self.head_type == "nondecreasing":
            self.base_head = torch.nn.Linear(head_input_dim, prediction_step)
            self.increment_head = torch.nn.Linear(
                head_input_dim, prediction_step * max(self.num_quantiles - 1, 0)
            )
        else:
            self.quantile_heads = torch.nn.ModuleList(
                torch.nn.Linear(head_input_dim, prediction_step)
                for _ in self.sorted_quantiles
            )
            
    def forward(self, x, current_feature=None):

        B, T, D = x.shape
        x_emb = self.input_linear(x)
        # (batch_size, window_len, model_dim)
        h, _ = self.rnn(x_emb)
        
        if current_feature != None:
            # (batch_size, window_len, model_dim+current_feature_dim)
            h = torch.cat([h, current_feature.repeat(1, T, 1)], dim=-1)

        if self.head_type == "independent":
            quantiles = torch.stack([head(h) for head in self.quantile_heads], dim=-1)
        else:
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

        if current_feature != None:
            # (batch_size, window_size, len(target_quantiles))
            out = model(x, current_feature=current_feature)
        else:
            # (batch_size, window_size, len(target_quantiles))
            out = model(x)
            
        return out[:, -1, :] # (batch_size, prediction_step * num_quantiles)
