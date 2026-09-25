"""Unit tests for latent space dimensionality reduction and manifold traversals."""

from pathlib import Path
import numpy as np
import torch

from src.configs.schema import load_config
from src.evaluation.latent_analysis import (
    plot_2d_latent_manifold,
    plot_latent_pca,
    plot_latent_tsne,
)
from src.evaluation.visualizer import (
    interpolate_synthetic_2d_grid,
    render_reconstruction_gallery,
    traverse_latent_dimensions,
)
from src.models.registry import build_vae_from_config


def test_plot_latent_tsne_and_pca_mock(tmp_path: Path):
    """Verify t-SNE and PCA plotting execute cleanly and save image files."""
    num_samples = 50
    latent_dim = 32

    mu_array = np.random.randn(num_samples, latent_dim)
    labels = np.random.randint(0, 10, size=num_samples)

    tsne_out = tmp_path / "tsne.png"
    pca_out = tmp_path / "pca.png"

    plot_latent_tsne(mu_array, labels, out_path=tsne_out, num_prior_samples=20)
    plot_latent_pca(mu_array, labels, out_path=pca_out, num_prior_samples=20)

    assert tsne_out.is_file()
    assert pca_out.is_file()
    assert tsne_out.stat().st_size > 0
    assert pca_out.stat().st_size > 0


def test_2d_latent_manifold_traversal(tmp_path: Path):
    """Verify 2D manifold coordinate meshgrid generation."""
    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)
    vae.eval()

    manifold_out = tmp_path / "manifold.png"
    result_path = plot_2d_latent_manifold(
        model=vae,
        dim_x=0,
        dim_y=1,
        grid_size=4,
        out_path=manifold_out,
        device=torch.device("cpu"),
        upscale=1,
    )

    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_synthetic_interpolation_and_sweeps(tmp_path: Path):
    """Verify pure synthetic prior interpolation and 1D axis sweeps."""
    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)
    vae.eval()

    synth_out = tmp_path / "synth_grid.png"
    sweep_out = tmp_path / "sweep.png"

    interpolate_synthetic_2d_grid(
        model=vae,
        grid_size=4,
        out_path=synth_out,
        device=torch.device("cpu"),
        upscale=1,
    )
    traverse_latent_dimensions(
        model=vae,
        dimensions=(0, 1),
        steps=5,
        out_path=sweep_out,
        device=torch.device("cpu"),
        upscale=1,
    )

    assert synth_out.is_file()
    assert sweep_out.is_file()


def test_render_reconstruction_gallery(tmp_path: Path):
    """Verify reconstruction gallery computes valid MSE and PSNR."""
    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)
    vae.eval()

    dummy_images = torch.randn(4, 3, 32, 32).clamp(-1.0, 1.0)
    gallery_out = tmp_path / "gallery.png"

    path, metrics = render_reconstruction_gallery(
        model=vae,
        images=dummy_images,
        out_path=gallery_out,
        device=torch.device("cpu"),
        upscale=1,
    )

    assert path.is_file()
    assert "gallery_mse" in metrics
    assert "gallery_psnr" in metrics
    assert metrics["gallery_mse"] >= 0.0
    assert metrics["gallery_psnr"] > 0.0
