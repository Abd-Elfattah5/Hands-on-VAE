"""Unit tests for checkpoint serialization, restoration, and schema portability."""

from pathlib import Path
import torch

from src.configs.schema import load_config
from src.models.registry import build_vae_from_config
from src.training.checkpoint import (
    load_checkpoint,
    restore_model_from_checkpoint,
    save_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path: Path):
    """Verify that saving and restoring a model checkpoint preserves weights and config."""
    cfg = load_config("configs/cifar10_baseline.yaml")
    model = build_vae_from_config(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "test_model.pt"

    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=150,
        config=cfg,
        metrics={"val_loss": 12.34},
        seed=42,
    )

    assert ckpt_path.is_file()

    # Load raw dictionary
    data = load_checkpoint(ckpt_path)
    assert data["format_version"] == "1.0"
    assert data["epoch"] == 3
    assert data["global_step"] == 150
    assert data["metrics"]["val_loss"] == 12.34
    assert data["provenance"]["seed"] == 42

    # Restore into fresh model
    restored_model, restored_data = restore_model_from_checkpoint(ckpt_path)
    assert isinstance(restored_model, torch.nn.Module)

    # Verify identical output on identical inputs
    x = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
    model.eval()
    restored_model.eval()

    with torch.no_grad():
        out_orig = model.encode(x, deterministic=True)
        out_restored = restored_model.encode(x, deterministic=True)

    assert torch.allclose(out_orig, out_restored, atol=1e-6)
