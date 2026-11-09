import math

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
    ``epanechnikov`` instead uses compactly supported chord-distance weights
    on unit-normalized representations, independently of temperature.
    """

    def __init__(self, encoded_rep, target, similarity_fn, temperature, device,
                 *, kernel="softmax", bandwidth=None):
        """
        Args
            encoded_rep: tensor, encoded representation for each timestep in the calibration set, (calib_size, dim)
            target: tensor, residual for each timestep, (calib_size, 1)
        """

        self.encoded_rep = encoded_rep
        self.target = target
        if kernel not in ("softmax", "epanechnikov"):
            raise ValueError("kernel must be 'softmax' or 'epanechnikov'.")
        if kernel == "epanechnikov" and similarity_fn != "cos_similarity":
            raise ValueError("Epanechnikov weighting requires cos_similarity.")
        self.kernel = kernel
        self.bandwidth = None
        self.weighting_fn = torch.nn.Softmax(dim=1)
        self.similarity_name = similarity_fn
        self.similarity_fn = SIM_FN_MAP[similarity_fn]
        self.device = device
        if kernel == "epanechnikov":
            message = "Epanechnikov weighting requires a finite positive scalar bandwidth."
            try:
                value = torch.as_tensor(bandwidth)
                if value.ndim != 0 or value.dtype == torch.bool or value.is_complex():
                    raise ValueError(message)
                self.bandwidth = float(bandwidth)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(message) from exc
            if not math.isfinite(self.bandwidth) or self.bandwidth <= 0:
                raise ValueError(message)
            # Temperature belongs only to the softmax estimator.
            self.temperature = None
            return
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

    @staticmethod
    def _unit_representations(representations):
        """Normalize nonzero finite vectors without an epsilon changing their norm."""
        if (
            representations.ndim != 2
            or representations.shape[1] == 0
            or not representations.is_floating_point()
            or not torch.isfinite(representations).all().item()
        ):
            raise ValueError("Epanechnikov representations must be finite real floating-point matrices.")
        scale = representations.abs().amax(dim=1, keepdim=True)
        if (scale == 0).any().item():
            raise ValueError("Epanechnikov representations must have nonzero norms.")
        # Scaling first avoids underflow for tiny vectors and overflow for large ones.
        scaled = representations / scale
        return scaled / torch.linalg.vector_norm(scaled, dim=1, keepdim=True)

    def compute_weights(self, query_rep):
        """Return normalized weights, or all zeros for an empty kernel neighborhood."""
        if self.kernel == "epanechnikov":
            queries = self._unit_representations(query_rep.to(self.device))
            calibration = self._unit_representations(self.encoded_rep.to(self.device))
            if queries.shape[1] != calibration.shape[1]:
                raise ValueError("query and key dimension must match")
            cosine = (queries @ calibration.T).clamp(-1.0, 1.0)
            squared_distance = (2.0 * (1.0 - cosine)).to(torch.float64)
            # Divide twice instead of squaring a very small or large bandwidth.
            scaled_distance = squared_distance / self.bandwidth / self.bandwidth
            unnormalized = (1.0 - scaled_distance).clamp_min(0.0)
            total = unnormalized.sum(dim=1, keepdim=True)
            denominator = torch.where(total > 0, total, torch.ones_like(total))
            return (unnormalized / denominator).to(queries.dtype)

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
        empty = probs.sum(dim=1) == 0 if self.kernel == "epanechnikov" else None
        sampling_probs = probs
        if empty is not None and empty.any().item():
            # A temporary point mass keeps the draw shape/RNG usage unchanged.
            # These rows' sampled residuals are replaced by zero below, implementing
            # the manuscript's empty-neighborhood CDF (a point mass at residual zero).
            sampling_probs = probs.clone()
            sampling_probs[empty, 0] = 1.0

        sampled_idx = torch.multinomial(sampling_probs, num_samples=sampling_num, replacement=True)
        if self.target.ndim == 2:
            target = self.target.squeeze(1).to(self.device)
        else:
            target = self.target.to(self.device)

        empirical = torch.gather(target.unsqueeze(0).expand(probs.shape[0], -1), dim=1, index=sampled_idx)
        if empty is not None:
            empirical[empty] = 0.0
        target_quantiles = torch.tensor(target_quantiles).to(self.device)

        return torch.quantile(empirical, target_quantiles, dim=1) # (len(target_quantiles), query_size)
