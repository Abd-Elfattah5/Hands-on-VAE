"""Heteroscedastic Gaussian observation likelihood with beta-NLL loss weighting."""

from typing import Any, Optional
import torch

from src.models.base import BaseLikelihood
from src.models.registry import register_likelihood


@register_likelihood("gaussian_hetero")
class HeteroscedasticGaussianLikelihood(BaseLikelihood):
    """Pixel-wise heteroscedastic Gaussian observation model stabilized via beta-NLL."""

    def __init__(self, beta_nll: float = 0.5) -> None:
        super().__init__()
        self.beta_nll = beta_nll

    def negative_log_likelihood(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        extra: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute pixel-wise beta-NLL loss per batch element.

        Formula:
            L_beta = 0.5 * sum_pixels [ (x - mu)^2 / sigma^2 + log(sigma^2) ] * (sigma^2)^beta_detached

        Args:
            reconstruction: Predicted mean tensor [B, C, H, W].
            target: Ground truth target image [B, C, H, W].
            extra: Auxiliary dictionary containing 'log_var_map'.

        Returns:
            Tensor of shape [B] containing loss per sample.
        """
        if extra is not None and "log_var_map" in extra:
            log_var = extra["log_var_map"]
        else:
            log_var = torch.zeros_like(reconstruction)

        var = torch.exp(log_var)
        sq_err = (reconstruction - target).pow(2)

        if self.beta_nll > 0.0:
            beta_weight = (var.detach()).pow(self.beta_nll)
            loss_per_pixel = 0.5 * (sq_err / var + log_var) * beta_weight
        else:
            loss_per_pixel = 0.5 * (sq_err / var + log_var)

        return torch.sum(loss_per_pixel, dim=[1, 2, 3])

    def sample(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Extract deterministic mean observation."""
        return reconstruction
