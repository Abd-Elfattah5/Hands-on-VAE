"""Type definitions and dataclasses for VAE inputs, outputs, and losses."""

from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class VAEOutput:
    """Strongly-typed forward pass return container for Variational Autoencoders."""

    reconstruction: torch.Tensor
    """Reconstructed image tensor, shape [B, C, H, W], bounded in [-1, 1]."""

    z: torch.Tensor
    """Sampled latent representation, shape [B, d]."""

    posterior: Any
    """Posterior distribution instance implementing BasePosteriorDistribution."""

    prior: Any
    """Prior distribution instance implementing BasePriorDistribution."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Auxiliary outputs (e.g. spatial variance map for heteroscedastic models)."""


@dataclass
class LossOutput:
    """Strongly-typed objective evaluation container for variational loss."""

    loss: torch.Tensor
    """Scalar loss tensor with active computational graph for backpropagation."""

    reconstruction_loss: torch.Tensor
    """Detached scalar value of negative log-likelihood / MSE reconstruction loss."""

    kl_divergence: torch.Tensor
    """Detached scalar value of analytical/sampled KL divergence."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Detailed decomposed metrics dictionary for logging (e.g. per-dimension KL)."""
