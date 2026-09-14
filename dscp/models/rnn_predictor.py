import torch
from utils.utils import generate_strided_feature


class RNNPredictor(torch.nn.Module):
    """
    stacked RNN for prediction
    """

    def __init__(
        self,
        rnn_type: str,
        dim_feature: int,
        dim_model: int,
        num_layer: int,
        prediction_step: int,
        current_feature_dim: int = 0,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super(RNNPredictor, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.dim_model = dim_model

        rnn_cls = {"rnn": torch.nn.RNN, "gru": torch.nn.GRU, "lstm": torch.nn.LSTM}[self.rnn_type]
        self.rnn = rnn_cls(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=num_layer,
            batch_first=batch_first,
            bidirectional=False,
            dropout=dropout if num_layer > 1 else 0.0,
        )

        # this will work as embedding layer for features
        self.input_linear = torch.nn.Linear(dim_feature, dim_model)
        self.output_linear = torch.nn.Linear(dim_model + current_feature_dim, prediction_step)  # no activation

    def forward(self, src, current_feature=None, return_repr=False):

        src_emb = self.input_linear(src)
        outputs, _ = self.rnn(src_emb)
        if current_feature is not None:
            outputs = torch.cat([outputs, current_feature.repeat(1, outputs.shape[1], 1)], dim=-1)

        if return_repr:
            return self.output_linear(outputs), outputs
        else:
            return self.output_linear(outputs)

    @staticmethod
    def encode(model, x, current_feature=None):

        x_emb = model.input_linear(x)
        outputs, _ = model.rnn(x_emb)
        if current_feature is not None:
            outputs = torch.cat([outputs, current_feature.repeat(1, outputs.shape[1], 1)], dim=-1)

        return outputs  # (B, T, D)

    @staticmethod
    def encode_dataloader(model, dataloader, strided_features, device, use_current_feature=False):

        repr_list = []
        residual_list = []
        for strided_x, strided_residual, strided_y, \
            target_x, target_residual, target_y, target_predictions in dataloader:

            strided_feature = generate_strided_feature(
                strided_x,
                strided_residual,
                strided_y,
                strided_features,
            )

            strided_feature = strided_feature.to(device)
            target_x = target_x.to(device)
            repr = model.encode(model,
                                strided_feature,
                                current_feature=target_x if use_current_feature else None)

            repr_list.append(repr[:, -1, :])
            residual_list.append(target_residual)

        return torch.cat(repr_list, dim=0), torch.cat(residual_list, dim=0)
