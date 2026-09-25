"""Dimensionality reduction and manifold analysis for 32D latent representations."""

from pathlib import Path
from typing import Optional, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from src.data.cifar10 import unnormalize
from src.evaluation.visualizer import upscale_tensor
from src.models.vae import VAE

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def extract_latent_embeddings(
    model: VAE,
    data_loader: DataLoader,
    device: torch.device,
    max_samples: int = 2500,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract posterior mean vectors mu(x) and ground-truth labels from a dataset partition.

    Args:
        model: Trained VAE model.
        data_loader: DataLoader containing image batches.
        device: Execution device.
        max_samples: Maximum number of samples to process.

    Returns:
        Tuple of (mu_array [N, latent_dim], labels_array [N]).
    """
    model.eval()
    model.to(device)

    all_mu: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    count = 0

    for batch in data_loader:
        images, labels = batch[0].to(device), batch[1]
        output = model(images)
        if hasattr(output.posterior, "mu"):
            mu = output.posterior.mu.cpu().numpy()
        else:
            mu = output.z.cpu().numpy()

        all_mu.append(mu)
        all_labels.append(labels.numpy())
        count += images.size(0)
        if count >= max_samples:
            break

    mu_array = np.concatenate(all_mu, axis=0)[:max_samples]
    labels_array = np.concatenate(all_labels, axis=0)[:max_samples]
    return mu_array, labels_array


def plot_latent_tsne(
    mu_array: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path = "artifacts/eval/latent_tsne.png",
    num_prior_samples: int = 500,
    random_state: int = 42,
) -> Path:
    """Project latent representations into 2D via t-SNE and render class-colored scatter plot."""
    latent_dim = mu_array.shape[1]

    # Combine posterior samples with standard Gaussian prior samples for alignment check
    prior_samples = np.random.randn(num_prior_samples, latent_dim)
    combined = np.concatenate([mu_array, prior_samples], axis=0)

    tsne = TSNE(n_components=2, perplexity=30.0, random_state=random_state, max_iter=1000)
    embedding = tsne.fit_transform(combined)

    mu_embedded = embedding[: len(mu_array)]
    prior_embedded = embedding[len(mu_array) :]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot prior samples in neutral background
    ax.scatter(
        prior_embedded[:, 0],
        prior_embedded[:, 1],
        c="lightgray",
        alpha=0.4,
        s=15,
        label="Prior N(0, I)",
        edgecolors="none",
    )

    # Plot data points colored by class
    cmap = plt.get_cmap("tab10")
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_idx
        ax.scatter(
            mu_embedded[mask, 0],
            mu_embedded[mask, 1],
            color=cmap(class_idx),
            alpha=0.7,
            s=20,
            label=class_name,
            edgecolors="none",
        )

    ax.set_title("t-SNE Latent Space Embedding (CIFAR-10 Posterior vs. Prior)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_latent_pca(
    mu_array: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path = "artifacts/eval/latent_pca.png",
    num_prior_samples: int = 500,
) -> Path:
    """Project latent representations into 2D via PCA and plot principal variance axes."""
    latent_dim = mu_array.shape[1]
    prior_samples = np.random.randn(num_prior_samples, latent_dim)

    pca = PCA(n_components=2)
    pca.fit(mu_array)

    mu_pca = pca.transform(mu_array)
    prior_pca = pca.transform(prior_samples)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    ax.scatter(
        prior_pca[:, 0],
        prior_pca[:, 1],
        c="lightgray",
        alpha=0.4,
        s=15,
        label="Prior N(0, I)",
        edgecolors="none",
    )

    cmap = plt.get_cmap("tab10")
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_idx
        ax.scatter(
            mu_pca[mask, 0],
            mu_pca[mask, 1],
            color=cmap(class_idx),
            alpha=0.7,
            s=20,
            label=class_name,
            edgecolors="none",
        )

    ax.set_title("PCA Latent Space Projection (CIFAR-10)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var1:.1f}% Variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% Variance)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_2d_latent_manifold(
    model: VAE,
    dim_x: int = 0,
    dim_y: int = 1,
    grid_size: int = 10,
    range_limit: float = 2.5,
    out_path: str | Path = "artifacts/eval/latent_manifold_2d.png",
    device: Optional[torch.device] = None,
    upscale: int = 4,
) -> Path:
    """Generate and render a continuous 2D generative manifold surface along two latent coordinates.

    All other latent coordinates are fixed at zero (the prior mean).

    Args:
        model: Trained VAE model.
        dim_x: Latent coordinate index along horizontal axis.
        dim_y: Latent coordinate index along vertical axis.
        grid_size: Number of steps per axis (total images = grid_size * grid_size).
        range_limit: Boundary for coordinate meshgrid [-range_limit, range_limit].
        out_path: Filepath to save rendered manifold PNG.
        device: Target execution device.
        upscale: Tile upscaling multiplier.

    Returns:
        Path to saved manifold grid artifact.
    """
    if device is None:
        device = next(model.parameters()).device

    latent_dim = getattr(model, "latent_dim", None)
    if latent_dim is None:
        latent_dim = model.prior.latent_dim

    model.eval()
    with torch.no_grad():
        x_vals = torch.linspace(-range_limit, range_limit, grid_size, device=device)
        y_vals = torch.linspace(range_limit, -range_limit, grid_size, device=device)  # top to bottom

        frames: list[torch.Tensor] = []
        for y_coord in y_vals:
            for x_coord in x_vals:
                z = torch.zeros(1, latent_dim, device=device)
                z[0, dim_x] = x_coord
                z[0, dim_y] = y_coord
                decoded = model.decode(z)
                frames.append(unnormalize(decoded))

        grid = torch.cat(frames, dim=0)
        if upscale > 1:
            grid = upscale_tensor(grid, factor=upscale)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        grid,
        str(out_file),
        nrow=grid_size,
        normalize=False,
        padding=2 * upscale,
    )

    return out_file
