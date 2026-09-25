"""Factorized diagonal Gaussian variational posterior distribution."""

from typing import Optional
import torch
import torch.nn as nn

from src.models.base import BasePosterior, BasePosteriorDistribution
from src.models.registry import register_posterior


class DiagonalGaussianDistribution(BasePosteriorDistribution):
    """Diagonal Gaussian variational posterior instance q_phi(z|x)."""

    def __init__(self, mu: torch.Tensor, log_var: torch.Tensor) -> None:
        self.mu = mu
        self.log_var = log_var
        self.sigma = torch.exp(0.5 * log_var)

    def rsample(self, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Differentiable sampling via the reparameterization trick: z = mu + sigma * eps."""
        if eps is None:
            eps = torch.randn_like(self.sigma)
        return self.mu + self.sigma * eps

    def sample(self) -> torch.Tensor:
        """Sample latent representation using the reparameterization trick."""
        return self.rsample()


@register_posterior("diagonal")
class DiagonalGaussianPosterior(BasePosterior):
    """Maps encoder feature representations into a diagonal Gaussian posterior."""

    def __init__(
        self,
        in_features: int,
        latent_dim: int = 32,
        clamp_min: float = -20.0,
        clamp_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_log_var = nn.Linear(in_features, latent_dim)

    def forward(self, features: torch.Tensor) -> DiagonalGaussianDistribution:
        """Predict mu and log_var and return a DiagonalGaussianDistribution."""
        mu = self.fc_mu(features)
        raw_log_var = self.fc_log_var(features)
        clamped_log_var = torch.clamp(raw_log_var, min=self.clamp_min, max=self.clamp_max)
        return DiagonalGaussianDistribution(mu=mu, log_var=clamped_log_var)
