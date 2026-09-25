"""Composite top-level Variational Autoencoder (VAE) module."""

from typing import Any, Optional
import torch
import torch.nn as nn

from src.models.base import (
    BaseDecoder,
    BaseEncoder,
    BaseLikelihood,
    BaseLoss,
    BasePosterior,
    BasePrior,
)
from src.models.types import LossOutput, VAEOutput


class VAE(nn.Module):
    """Composite Variational Autoencoder assembling modular subcomponents."""

    def __init__(
        self,
        encoder: BaseEncoder,
        posterior: BasePosterior,
        prior: BasePrior,
        decoder: BaseDecoder,
        likelihood: BaseLikelihood,
        loss_fn: BaseLoss,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.posterior = posterior
        self.prior = prior
        self.decoder = decoder
        self.likelihood = likelihood
        self.loss_fn = loss_fn
        self.config = config or {}

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """Execute forward pass mapping input observation to reconstructed output.

        Args:
            x: Input observation tensor [B, C, H, W].

        Returns:
            VAEOutput container with reconstruction, latent sample, posterior, and prior.
        """
        features = self.encoder(x)
        posterior_dist = self.posterior(features)
        z = posterior_dist.sample()
        decoder_output = self.decoder(z)

        if isinstance(decoder_output, tuple):
            reconstruction, log_var_map = decoder_output
            extra = {"log_var_map": log_var_map}
        else:
            reconstruction = decoder_output
            extra = {}

        return VAEOutput(
            reconstruction=reconstruction,
            z=z,
            posterior=posterior_dist,
            prior=self.prior,
            extra=extra,
        )

    def compute_loss(self, output: VAEOutput, target: torch.Tensor) -> LossOutput:
        """Compute training objective and decomposed metrics."""
        return self.loss_fn(output, target)

    def encode(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Encode an input observation into latent space.

        Args:
            x: Input observation tensor [B, C, H, W].
            deterministic: If True, returns posterior mean mu; otherwise samples z.

        Returns:
            Latent representation tensor [B, latent_dim].
        """
        features = self.encoder(x)
        posterior_dist = self.posterior(features)
        if deterministic and hasattr(posterior_dist, "mu"):
            return posterior_dist.mu
        return posterior_dist.sample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into observation parameters.

        Args:
            z: Latent tensor [B, latent_dim].

        Returns:
            Observation tensor [B, C, H, W] in [-1, 1].
        """
        out = self.decoder(z)
        if isinstance(out, tuple):
            return out[0]
        return out

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Synthesize novel samples by drawing latent vectors from the prior.

        Args:
            num_samples: Number of samples to synthesize.
            device: Target torch device.

        Returns:
            Generated image tensor [num_samples, C, H, W] in [-1, 1].
        """
        z = self.prior.sample(num_samples, device=device)
        return self.decode(z)
