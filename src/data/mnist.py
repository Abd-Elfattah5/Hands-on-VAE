"""MNIST dataset loading, transforms, and dataloaders."""

from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset, default_collate, random_split
import torchvision
import torchvision.transforms as transforms


def get_mnist_transforms() -> transforms.Compose:
    """Return image transforms padding 28x28 to 32x32 and normalizing [-1, 1]."""
    return transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])


def get_mnist_datasets(
    data_dir: str | Path = "data",
    val_split: float = 0.1,
    seed: int = 42,
    download: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load MNIST dataset and perform deterministic train/val splitting.

    Args:
        data_dir: Root directory for MNIST storage.
        val_split: Fraction of train partition to reserve for validation (e.g. 0.1 = 10%).
        seed: Random seed for deterministic data splitting.
        download: Whether to download if not present locally.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    transform = get_mnist_transforms()

    full_train = torchvision.datasets.MNIST(
        root=str(data_path),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=str(data_path),
        train=False,
        download=download,
        transform=transform,
    )

    total_len = len(full_train)  # 60,000
    val_len = int(total_len * val_split)  # 6,000
    train_len = total_len - val_len  # 54,000

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train, [train_len, val_len], generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def get_mnist_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    download: bool = True,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for MNIST train, validation, and test splits.

    Args:
        data_dir: Storage directory.
        batch_size: Batch size.
        val_split: Fraction of train data for validation.
        num_workers: Worker count for data loading.
        seed: Random seed.
        download: Whether to download dataset.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds, val_ds, test_ds = get_mnist_datasets(
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
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
