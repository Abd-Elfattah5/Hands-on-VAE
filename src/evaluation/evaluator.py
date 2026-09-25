"""Quantitative model evaluation on test partitions."""

from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    variance_threshold: float = 0.01,
) -> dict[str, Any]:
    """Perform rigorous quantitative evaluation on a data partition.

    Computes:
        - Mean total ELBO loss
        - Reconstruction negative log-likelihood
        - Analytical KL divergence
        - Mean Squared Error (MSE)
        - Active latent units count: dimensions where Var_x(E_q[z_j]) > threshold

    Args:
        model: Trained VAE model instance.
        data_loader: Evaluation data loader (e.g. CIFAR-10 test split).
        device: Target execution device.
        variance_threshold: Minimum variance across dataset to consider a latent unit active.

    Returns:
        Dictionary of computed quantitative metrics.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_mse = 0.0
    num_samples = 0

    all_mu: list[torch.Tensor] = []

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        images = batch[0].to(device)
        batch_size = images.size(0)

        output = model(images)
        loss_out = model.compute_loss(output, images)

        total_loss += loss_out.loss.item() * batch_size
        total_recon += loss_out.reconstruction_loss.item() * batch_size
        total_kl += loss_out.kl_divergence.item() * batch_size

        batch_mse = torch.mean((output.reconstruction - images) ** 2).item()
        total_mse += batch_mse * batch_size

        if hasattr(output.posterior, "mu"):
            all_mu.append(output.posterior.mu.cpu())

        num_samples += batch_size

    mean_loss = total_loss / num_samples
    mean_recon = total_recon / num_samples
    mean_kl = total_kl / num_samples
    mean_mse = total_mse / num_samples

    # Compute active latent dimensions: Var_x(mu_j) > variance_threshold
    active_units = 0
    if all_mu:
        concat_mu = torch.cat(all_mu, dim=0)  # [N, d]
        var_per_dim = torch.var(concat_mu, dim=0)  # [d]
        active_units = int(torch.sum(var_per_dim > variance_threshold).item())

    return {
        "test_loss": mean_loss,
        "test_recon": mean_recon,
        "test_kl": mean_kl,
        "test_mse": mean_mse,
        "active_units": active_units,
        "total_samples": num_samples,
    }
