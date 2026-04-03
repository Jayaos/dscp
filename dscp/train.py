import torch
import numpy as np


def compute_quantile_loss(outputs, targets, desired_quantiles):
    """
    This function compute the quantile loss a.k.a. pinball loss separately per each sample, time-step, and quantile.

    Parameters
    ----------
    outputs: torch.Tensor
        The outputs of the model (num_prediction_step * batch_size * num_quantiles)
    targets: torch.Tensor
        The observed target for each horizon (num_prediction_step * batch_size)
    desired_quantiles: torch.Tensor
        A tensor representing the desired quantiles, of shape (num_quantiles)

    Returns
    -------
    losses_array: torch.Tensor
        a tensor [num_samples x num_horizons x num_quantiles] containing the quantile loss for each sample,time-step and
        quantile.
    """

    # compute the actual error between the observed target and each predicted quantile
    # TODO: consider mask in resid_y 
    # errors = targets.reshape((targets.size()[0]*targets.size()[1],1)) - outputs.reshape((outputs.size()[0]*outputs.size()[1],outputs.size()[2])) 
    errors = targets.unsqueeze(-1) - outputs # (num_samples * num_time_steps * num_quantiles)

    # compute the loss separately for each sample,time-step,quantile
    losses_array = torch.max((desired_quantiles - 1) * errors, desired_quantiles * errors) # element-wise max

    # sum losses over quantiles and average across time and observations: scalar
    return (losses_array.sum(dim=-1)).mean(dim=-1).mean()


def train_quantile_prediction_model(model, train_dataloader, valid_dataloader, max_epoch, 
                                    additional_training_epoch, 
                                    learning_rate, early_stop, device="cpu"):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_loss_record = []
    validation_loss_record = []
    best_loss = np.inf
    best_epoch = 0

    for e in range(max_epoch):

        if early_stop:
            if (e+1-best_epoch) >= early_stop:
                break

        batch_loss_sum = 0.
        batch_num = len(train_dataloader)

    ...

            if shifted_concatenation:
                # concat [x_{t-k}, y_{t-k}, \epsilon_{t-k-1}] ... to predict \epsilon_{t}
                strided_x = to_strided_feature(heldout_x, 
                                               past_window, 
                                               prediction_steps-1) 
                strided_x = strided_x[1:]
                strided_residual, target_residual = to_strided_residual(heldout_residuals, 
                                                                        past_window, 
                                                                        prediction_steps)
                # normalized strided_y
                strided_y, _ = to_strided_residual(heldout_y.flatten(), 
                                                  past_window, 
                                                  prediction_steps)
                # raw target_y, not normalized
                _, target_y = to_strided_residual(item["heldout_y"].flatten(), 
                                                  past_window, 
                                                  prediction_steps)
                # raw target_predictions, not normalized
                _, target_predictions = to_strided_residual(item["heldout_predictions"].flatten(), 
                                                        past_window, 
                                                        prediction_steps)