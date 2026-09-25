"""Abstract Base Classes for modular VAE components."""

from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn

from src.models.types import LossOutput, VAEOutput


class BasePosteriorDistribution(ABC):
    """Interface for posterior distribution instances returned by BasePosterior."""

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """Sample latent vector using reparameterization."""
        pass


class BasePriorDistribution(ABC):
    """Interface for prior distribution instances."""

    @abstractmethod
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample latent vectors from the prior."""
        pass


class BaseEncoder(nn.Module, ABC):
    """Encodes an input observation tensor [B, C, H, W] into feature representations."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns feature tensor [B, hidden_dim]."""
        pass

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Number of output features provided to posterior projection."""
        pass


class BasePosterior(nn.Module, ABC):
    """Maps encoder feature representations into a variational posterior distribution."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> Any:
        """Returns a PosteriorDistribution instance supporting .sample() and distribution parameters."""
        pass


class BasePrior(nn.Module, ABC):
    """Represents the latent prior distribution p(z)."""

    @abstractmethod
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Draws latent vectors directly from the prior [num_samples, latent_dim]."""
        pass

    @abstractmethod
    def kl_divergence(self, posterior_dist: Any) -> torch.Tensor:
        """Computes or estimates D_KL(posterior || prior). Returns tensor [B]."""
        pass


class BaseDecoder(nn.Module, ABC):
    """Maps latent vector z [B, latent_dim] into likelihood distribution parameters."""

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns raw reconstruction parameter tensor [B, out_channels, H, W]."""
        pass


class BaseLikelihood(nn.Module, ABC):
    """Computes log-likelihood log p_theta(x|z) given decoder output and targets."""

    @abstractmethod
    def negative_log_likelihood(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        extra: Any = None,
    ) -> torch.Tensor:
        """Returns negative log-likelihood per batch element, shape [B]."""
        pass

    @abstractmethod
    def sample(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Extracts deterministic mean or samples observation from the distribution."""
        pass


class BaseLoss(nn.Module, ABC):
    """Computes total training objective from VAEOutput and target observations."""

    @abstractmethod
    def forward(self, output: VAEOutput, target: torch.Tensor) -> LossOutput:
        """Calculates scalar loss and decomposed metrics."""
        pass
