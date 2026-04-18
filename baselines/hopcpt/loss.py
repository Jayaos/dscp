import torch


def compute_hopfield_net_loss(hopfield_net, memory_feature, memory_value, absolute_residual=True):
    """
    MSE loss in hopcpt paper

    :param memory_feature: (batch_size, memory_length, dim_feature)
    :param memory_value: (batch_size, memory_length, 1)
    """

    if memory_value.dim() == 1:
        memory_value = memory_value.unsqueeze(0)

    if memory_value.dim() == 2:
        memory_value = memory_value.unsqueeze(-1)

    # memory feature is used for query feature as well to compute the loss
    association_mask = get_association_mask(memory_feature)
    preds = hopfield_net(memory_feature, memory_value, association_mask=association_mask) # (batch_size, memory_length, 1)
    if absolute_residual:
        preds = preds.abs()
        memory_value = memory_value.abs()
    loss = torch.nn.functional.mse_loss(preds, memory_value, reduction="none").sum()

    return loss


def get_association_mask(memory_feature):
    """
    assuming batch size is 1
    """

    device = memory_feature.device

    batch_size, memory_length, feature_dim = memory_feature.shape
    mask = torch.eye(memory_length, dtype=torch.bool, device=device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, memory_length, memory_length)

    return mask


def return_association_matrix(hopfield_net, memory_feature):

    device = memory_feature.device
    batch_size, memory_length, _ = memory_feature.shape

    # get_association_matrix() does not need value to compute the association matrix but needs value as an argument
    # this is dummy value
    v = torch.zeros((batch_size, memory_length, 1)).to(device)

    association_mask = get_association_mask(memory_feature)


    # encoding
    encoded_k = hopfield_net.context_encoder(memory_feature) # (batch_size, memory_length, dim_context_encoding)
    encoded_k = hopfield_net._append_temporal_encoding(encoded_k, memory_length)

    return hopfield_net.hopfield_net.get_association_matrix((encoded_k, encoded_k, v), association_mask=association_mask) # (batch_size, 1, 1, memory_length)
