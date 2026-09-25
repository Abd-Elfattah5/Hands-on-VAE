"""Standard isotropic Gaussian prior distribution p(z) = N(0, I)."""

from typing import Any
import torch

from src.models.base import BasePrior
from src.models.registry import register_prior


@register_prior("standard_gaussian")
class StandardGaussianPrior(BasePrior):
    """Standard isotropic normal prior distribution N(0, I)."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample latent vectors directly from standard normal distribution."""
        return torch.randn(num_samples, self.latent_dim, device=device)

    def kl_divergence(self, posterior_dist: Any) -> torch.Tensor:
        """Compute exact closed-form analytical KL divergence D_KL(q_phi(z|x) || p(z)).

        Formula:
            D_KL = -0.5 * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)

        Args:
            posterior_dist: Posterior distribution object with mu and log_var attributes.

        Returns:
            Tensor of shape [B] representing KL divergence per sample.
        """
        mu = posterior_dist.mu
        log_var = posterior_dist.log_var
        # Analytical KL formula summed across latent dimensions
        kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - torch.exp(log_var))
        return torch.sum(kl_per_dim, dim=1)
