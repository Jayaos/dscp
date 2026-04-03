import torch
from utils.utils import dot_product, cos_similarity


SIM_FN_MAP = {
    "dot_product": dot_product,
    "cos_similarity": cos_similarity,
}


class CDFApproximation():
    """
    class to approximate cdf using weight
    """

    def __init__(self, encoded_rep, target, similarity_fn, temperature, device):
        """
        :param encoded_rep: encoded representation for each timestep in the calibration set, (calib_size, dim)
        :param target: residual for each timestep, (calib_size, )
        """

        self.encoded_rep = encoded_rep
        self.target = target
        self.weighting_fn = torch.nn.Softmax(dim=1)
        self.similarity_fn = SIM_FN_MAP[similarity_fn]
        self.temperature = torch.tensor(temperature)
        self.device = device

    def approximate_quantile(self, query_rep, target_quantiles, sampling_num):
        """
        :param query_rep: (query_batch_size, dim_model)
        :param target_quantiles: 
        :param sampling_num: 
        """

        o = self.similarity_fn(query_rep.to(self.device), self.encoded_rep.to(self.device)) # (query_size, calib_size)
        probs = self.weighting_fn(self.temperature.to(self.device)*o) # (query_size, calib_size)

        sampled_idx = torch.multinomial(probs, num_samples=sampling_num, replacement=True) # (query_size, sampling_num)
        # (query_size, sampling_num)
        if self.target.ndim == 2:
            target = self.target.squeeze(1).to(self.device)
        else:
            target = self.target.to(self.device)

        empirical = torch.gather(target.unsqueeze(0).expand(probs.shape[0], -1), dim=1, index=sampled_idx) 
        target_quantiles = torch.tensor(target_quantiles).to(self.device)

        return torch.quantile(empirical, target_quantiles, dim=1) # (len(target_quantiles), query_size)