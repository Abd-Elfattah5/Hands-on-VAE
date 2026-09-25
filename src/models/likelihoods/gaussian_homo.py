"""Homoscedastic Gaussian observation likelihood."""

import math
import torch

from src.models.base import BaseLikelihood
from src.models.registry import register_likelihood


@register_likelihood("gaussian_homo")
class HomoscedasticGaussianLikelihood(BaseLikelihood):
    """Homoscedastic continuous Gaussian observation model with fixed variance."""

    def __init__(self, fixed_sigma: float = 1.0, include_constant: bool = False) -> None:
        super().__init__()
        self.fixed_sigma = fixed_sigma
        self.variance = fixed_sigma ** 2
        self.include_constant = include_constant

    def negative_log_likelihood(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log-likelihood per batch element.

        Args:
            reconstruction: Model output mean tensor [B, C, H, W].
            target: Ground-truth target image tensor [B, C, H, W].

        Returns:
            Tensor of shape [B] containing NLL per sample.
        """
        # Sum of squared errors across spatial dimensions (C, H, W)
        sse = torch.sum((reconstruction - target).pow(2), dim=[1, 2, 3])
        nll = sse / (2.0 * self.variance)

        if self.include_constant:
            num_pixels = target[0].numel()
            const = 0.5 * num_pixels * math.log(2.0 * math.pi * self.variance)
            nll = nll + const

        return nll

    def sample(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Extract deterministic mean observation."""
        return reconstruction
