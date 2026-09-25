"""Evidence Lower Bound (ELBO) variational loss objective."""

import torch

from src.models.base import BaseLikelihood, BaseLoss
from src.models.registry import register_loss
from src.models.types import LossOutput, VAEOutput


@register_loss("elbo")
class ELBOLoss(BaseLoss):
    """Analytical Evidence Lower Bound objective: NLL + beta * KL."""

    def __init__(self, likelihood: BaseLikelihood, beta: float = 1.0) -> None:
        super().__init__()
        self.likelihood = likelihood
        self.beta = beta

    def forward(self, output: VAEOutput, target: torch.Tensor) -> LossOutput:
        """Compute decomposed ELBO loss from VAEOutput and target images.

        Args:
            output: Container with reconstruction, posterior, and prior objects.
            target: Ground-truth target image tensor [B, C, H, W].

        Returns:
            LossOutput containing scalar loss and detached diagnostic components.
        """
        # Reconstruction negative log-likelihood per sample [B]
        nll_per_sample = self.likelihood.negative_log_likelihood(output.reconstruction, target)

        # Analytical KL divergence per sample [B]
        kl_per_sample = output.prior.kl_divergence(output.posterior)

        # Objective per sample: NLL + beta * KL
        total_per_sample = nll_per_sample + self.beta * kl_per_sample

        # Mean across batch for gradient optimization
        loss = torch.mean(total_per_sample)
        recon_loss = torch.mean(nll_per_sample).detach()
        kl_div = torch.mean(kl_per_sample).detach()

        # Additional diagnostic metrics
        with torch.no_grad():
            raw_mse = torch.mean((output.reconstruction - target) ** 2).item()

        metrics = {
            "total_loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_div": kl_div.item(),
            "mse": raw_mse,
        }

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_divergence=kl_div,
            metrics=metrics,
        )
