"""Unit tests for sample synthesis and deterministic prior generation."""

from pathlib import Path
import torch

from src.configs.schema import load_config
from src.evaluation.visualizer import generate_sample_grid
from src.models.registry import build_vae_from_config


def test_deterministic_generation(tmp_path: Path):
    """Verify that identical random seeds produce bitwise identical sample grids."""
    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)
    vae.eval()

    grid1 = tmp_path / "grid1.png"
    grid2 = tmp_path / "grid2.png"

    generate_sample_grid(vae, num_samples=16, out_path=grid1, seed=12345, device=torch.device("cpu"))
    generate_sample_grid(vae, num_samples=16, out_path=grid2, seed=12345, device=torch.device("cpu"))

    with open(grid1, "rb") as f1, open(grid2, "rb") as f2:
        assert f1.read() == f2.read(), "Generated grids with identical seeds should be bit-for-bit identical"


def test_sample_shapes_and_bounds():
    """Verify sampled images adhere to [num_samples, 3, 32, 32] and [-1, 1]."""
    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)
    vae.eval()

    samples = vae.sample(num_samples=4, device=torch.device("cpu"))
    assert samples.shape == (4, 3, 32, 32)
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0
