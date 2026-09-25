"""Checkpoint serialization, restoration, and provenance tracking."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn

from src.models.registry import build_vae_from_config


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    config: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> Path:
    """Serialize model checkpoint with complete metadata and provenance.

    Args:
        path: Destination filepath for the .pt checkpoint.
        model: VAE model instance.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler instance.
        epoch: Completed epoch index.
        global_step: Cumulative training step index.
        config: Serialized experiment configuration.
        metrics: Validation and training metrics dictionary.
        seed: Random seed used for the experiment.

    Returns:
        Path to the saved checkpoint file.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data: dict[str, Any] = {
        "format_version": "1.0",
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config or getattr(model, "config", {}),
        "metrics": metrics or {},
        "provenance": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
        },
    }

    torch.save(checkpoint_data, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load raw checkpoint dictionary from disk.

    Args:
        path: Path to .pt checkpoint file.
        map_location: Target device for loaded tensors.

    Returns:
        Checkpoint dictionary adhering to schema.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format in {ckpt_path}: expected dictionary")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint in {ckpt_path} is missing 'model_state_dict'")

    return checkpoint


def restore_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    model: Optional[nn.Module] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Instantiate and restore model weights from a serialized checkpoint.

    Args:
        path: Path to .pt checkpoint file.
        map_location: Target device for restored model.
        model: Optional pre-constructed model instance to load state into.

    Returns:
        Tuple of (restored_model, checkpoint_dict).
    """
    checkpoint = load_checkpoint(path, map_location=map_location)

    if model is None:
        config = checkpoint.get("config", {})
        if not config:
            raise ValueError(f"Checkpoint {path} does not contain 'config' required to instantiate model")
        model = build_vae_from_config(config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()

    return model, checkpoint
