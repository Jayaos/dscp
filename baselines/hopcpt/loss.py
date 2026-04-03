import torch


def compute_hopfield_net_loss(hopfield_net, memory_feature, memory_value):
    """
    MSE loss in hopcpt paper

    :param memory_feature: (batch_size, memory_length, dim_feature)
    :param memory_value: (batch_size, memory_length, 1)
    """

    if memory_value.dim() == 1:
        memory_value.unsqueeze_(0) 

    if memory_value.dim() == 2:
        memory_value.unsqueeze_(-1) 

    # memory feature is used for query feature as well to compute the loss
    association_mask = get_association_mask(memory_feature)
    preds = hopfield_net(memory_feature, memory_value, association_mask=association_mask) # (batch_size, memory_length, 1)
    preds = preds.abs()
    e_abs = memory_value.abs() # memory value is used as target value to compute loss
    loss = torch.nn.functional.mse_loss(preds, e_abs, reduce=False).sum()

    return loss


def get_association_mask(memory_feature):
    """
    assuming batch size is 1
    """

    device = memory_feature.device

    batch_size, memory_length, feature_dim = memory_feature.shape
    mask = torch.eye(memory_length, dtype=torch.bool).unsqueeze(0).to(device) # (1, memory_length, memory_length)

    return mask


def return_association_matrix(hopfield_net, memory_feature):

    device = memory_feature.device
    batch_size, memory_length, _ = memory_feature.shape

    # get_association_matrix() does not need value to compute the association matrix but needs value as an argument
    # this is dummy value
    v = torch.zeros((batch_size, memory_length, 1)).to(device)

    association_mask = get_association_mask(memory_feature)


    # temporal encoding t/T
    t = torch.arange(memory_length, device=device)
    t = t / memory_length # (memory_length,)
    t = t.view(1, memory_length, 1) # (1, memory_length, 1)
    t = t.expand(batch_size, -1, -1) # (batch_size, memory_length, 1)

    # encoding
    encoded_k = hopfield_net.context_encoder(memory_feature) # (batch_size, memory_length, dim_context_encoding)
    encoded_k = torch.cat([encoded_k, t], dim=-1) # (batch_size, memory_length, dim_context_encoding + 1)

    return hopfield_net.hopfield_net.get_association_matrix((encoded_k, encoded_k, v), association_mask=association_mask) # (batch_size, 1, 1, memory_length)