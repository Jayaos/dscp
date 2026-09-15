import torch
from utils.utils import dot_product, cos_similarity, negative_squared_euclidean


SIM_FN_MAP = {
    "dot_product": dot_product,
    "cos_similarity": cos_similarity,
    "euclidean": negative_squared_euclidean,
}


class LocalConformalPrediction():
    """
    Approximate residual quantiles using representation-weighted calibration samples.

    ``euclidean`` uses softmax(-squared_distance / temperature).
    Dot product and cosine retain softmax(temperature * similarity).
    """

    def __init__(self, encoded_rep, target, similarity_fn, temperature, device):
        """
        Args
            encoded_rep: tensor, encoded representation for each timestep in the calibration set, (calib_size, dim)
            target: tensor, residual for each timestep, (calib_size, 1)
        """

        self.encoded_rep = encoded_rep
        self.target = target
        self.weighting_fn = torch.nn.Softmax(dim=1)
        self.similarity_name = similarity_fn
        self.similarity_fn = SIM_FN_MAP[similarity_fn]
        try:
            self.temperature = torch.tensor(temperature)
        except (TypeError, ValueError, RuntimeError) as exc:
            if similarity_fn == "euclidean":
                raise ValueError(
                    "Euclidean weighting requires a finite positive scalar temperature."
                ) from exc
            raise
        if similarity_fn == "euclidean" and (
            self.temperature.ndim != 0
            or self.temperature.dtype == torch.bool
            or self.temperature.is_complex()
            or not torch.isfinite(self.temperature).item()
            or self.temperature.item() <= 0
        ):
            raise ValueError("Euclidean weighting requires a finite positive scalar temperature.")
        self.device = device

    def compute_weights(self, query_rep):
        """Return normalized weights with shape (query_batch_size, calibration_size)."""
        scores = self.similarity_fn(
            query_rep.to(self.device), self.encoded_rep.to(self.device)
        )
        temperature = self.temperature.to(self.device)
        if self.similarity_name == "euclidean":
            # Center before division so a very small temperature cannot turn
            # every logit into -inf. The shared shift leaves softmax unchanged.
            scores = (scores - scores.max(dim=1, keepdim=True).values) / temperature
        else:
            scores = temperature * scores
        return self.weighting_fn(scores)

    def approximate_quantile(self, query_rep, target_quantiles, sampling_num):
        """
        :param query_rep: (query_batch_size, dim_model)
        :param target_quantiles: 
        :param sampling_num: 
        """

        probs = self.compute_weights(query_rep) # (query_size, calib_size)

        sampled_idx = torch.multinomial(probs, num_samples=sampling_num, replacement=True) # (query_size, sampling_num)
        # (query_size, sampling_num)
        if self.target.ndim == 2:
            target = self.target.squeeze(1).to(self.device)
        else:
            target = self.target.to(self.device)

        empirical = torch.gather(target.unsqueeze(0).expand(probs.shape[0], -1), dim=1, index=sampled_idx) 
        target_quantiles = torch.tensor(target_quantiles).to(self.device)

        return torch.quantile(empirical, target_quantiles, dim=1) # (len(target_quantiles), query_size)
