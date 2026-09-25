"""Shared pytest fixtures and synthetic data generators for VAE test suite."""

import pytest
import torch


@pytest.fixture
def device():
    """Device for testing (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def synthetic_image_batch():
    """Synthetic CIFAR-10 image batch [4, 3, 32, 32] bounded in [-1, 1]."""
    # Generate random floats in [-1.0, 1.0]
    return torch.rand(4, 3, 32, 32) * 2.0 - 1.0


@pytest.fixture
def synthetic_latent_batch():
    """Synthetic latent representation batch [4, 32]."""
    return torch.randn(4, 32)
