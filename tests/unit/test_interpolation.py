"""Unit tests for latent manifold interpolation mathematics."""

from pathlib import Path
import torch

from src.configs.schema import load_config
from src.evaluation.visualizer import interpolate_2d_grid, interpolate_images, lerp, slerp
from src.models.registry import build_vae_from_config


def test_lerp_endpoints():
    """Verify linear interpolation at alpha=0.0 and alpha=1.0 returns endpoints."""
    z1 = torch.tensor([[1.0, 2.0, 3.0]])
    z2 = torch.tensor([[5.0, 6.0, 7.0]])

    interp_start = lerp(0.0, z1, z2)
    interp_end = lerp(1.0, z1, z2)
    interp_mid = lerp(0.5, z1, z2)

    assert torch.allclose(interp_start, z1)
    assert torch.allclose(interp_end, z2)
    assert torch.allclose(interp_mid, torch.tensor([[3.0, 4.0, 5.0]]))


def test_slerp_endpoints():
    """Verify spherical linear interpolation returns normalized endpoints at bounds."""
    z1 = torch.tensor([[1.0, 0.0, 0.0]])
    z2 = torch.tensor([[0.0, 1.0, 0.0]])

    interp_start = slerp(0.0, z1, z2)
    interp_end = slerp(1.0, z1, z2)

    assert torch.allclose(interp_start, z1, atol=1e-5)
    assert torch.allclose(interp_end, z2, atol=1e-5)


def test_interpolate_images_execution(tmp_path: Path):
    """Verify end-to-end latent interpolation strip generation."""
    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)
    vae.eval()

    img1 = torch.randn(1, 3, 32, 32).clamp(-1.0, 1.0)
    img2 = torch.randn(1, 3, 32, 32).clamp(-1.0, 1.0)

    out_strip = tmp_path / "strip.png"
    result_path = interpolate_images(
        model=vae,
        img1=img1,
        img2=img2,
        steps=5,
        out_path=out_strip,
        device=torch.device("cpu"),
    )

    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_interpolate_2d_grid_execution(tmp_path: Path):
    """Verify end-to-end 4-corner 2D latent interpolation grid generation."""
    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)
    vae.eval()

    img1 = torch.randn(1, 3, 32, 32).clamp(-1.0, 1.0)
    img2 = torch.randn(1, 3, 32, 32).clamp(-1.0, 1.0)
    img3 = torch.randn(1, 3, 32, 32).clamp(-1.0, 1.0)
    img4 = torch.randn(1, 3, 32, 32).clamp(-1.0, 1.0)

    out_grid = tmp_path / "grid_2d.png"
    result_path = interpolate_2d_grid(
        model=vae,
        img_tl=img1,
        img_tr=img2,
        img_bl=img3,
        img_br=img4,
        grid_size=4,
        out_path=out_grid,
        device=torch.device("cpu"),
    )

    assert result_path.is_file()
    assert result_path.stat().st_size > 0
