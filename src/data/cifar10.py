"""CIFAR-10 data pipeline, transforms, and dataloaders."""

from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset, default_collate, random_split
import torchvision
import torchvision.transforms as transforms


def get_cifar10_transforms() -> transforms.Compose:
    """Return image normalization transform mapping [0, 255] pixels to [-1, 1]."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse transform converting [-1, 1] normalized tensors back to [0, 1] image range."""
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)


def vae_collate_fn(batch: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate batch and verify target observation values reside strictly within [-1, 1]."""
    collated = default_collate(batch)
    images, targets = collated[0], collated[1]
    # Allow numerical tolerance for floating point representations
    if images.min() < -1.01 or images.max() > 1.01:
        raise ValueError(
            f"Observation batch pixel values outside [-1, 1] bounds: "
            f"min={images.min().item():.4f}, max={images.max().item():.4f}"
        )
    return images, targets


def get_cifar10_datasets(
    data_dir: str | Path = "data",
    val_split: float = 0.1,
    seed: int = 42,
    download: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load CIFAR-10 datasets and perform deterministic train/val split.

    Args:
        data_dir: Root directory for downloading and storing CIFAR-10.
        val_split: Fraction of training dataset reserved for validation (e.g. 0.1 for 10%).
        seed: Random seed for deterministic train/val splitting.
        download: Whether to download CIFAR-10 if not present locally.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    transform = get_cifar10_transforms()

    full_train = torchvision.datasets.CIFAR10(
        root=str(data_path),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_path),
        train=False,
        download=download,
        transform=transform,
    )

    total_len = len(full_train)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train, [train_len, val_len], generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def get_cifar10_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    download: bool = True,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for CIFAR-10 train, validation, and test splits.

    Args:
        data_dir: Root directory for storing CIFAR-10 data.
        batch_size: Mini-batch size for training and evaluation.
        val_split: Fraction of training data to use for validation.
        num_workers: Number of subprocesses for data loading.
        seed: Random seed for deterministic data splitting.
        download: Whether to download CIFAR-10 if not found locally.
        pin_memory: Pin memory for faster GPU transfer (auto-detected if None).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds, val_ds, test_ds = get_cifar10_datasets(
        data_dir=data_dir,
        val_split=val_split,
        seed=seed,
        download=download,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=vae_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=vae_collate_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=vae_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
