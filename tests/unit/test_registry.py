"""Unit tests for component registry lookup and dynamic model building."""

import pytest
import torch

from src.configs.schema import load_config
from src.models.base import BaseEncoder
from src.models.registry import (
    ENCODER_REGISTRY,
    build_vae_from_config,
    get_decoder,
    get_encoder,
    get_likelihood,
    get_loss,
    get_posterior,
    get_prior,
)


def test_registry_lookup():
    """Verify registered components can be resolved by identifier."""
    # Ensure baseline components are imported and registered
    import src.models.encoders.cnn
    import src.models.decoders.cnn
    import src.models.posteriors.diagonal
    import src.models.priors.standard_gaussian
    import src.models.likelihoods.gaussian_homo
    import src.models.losses.elbo

    assert get_encoder("cnn") is not None
    assert get_decoder("cnn") is not None
    assert get_posterior("diagonal") is not None
    assert get_prior("standard_gaussian") is not None
    assert get_likelihood("gaussian_homo") is not None
    assert get_loss("elbo") is not None


def test_build_vae_from_config():
    """Verify dynamic construction of composite VAE module from YAML config."""
    import src.models.encoders.cnn
    import src.models.decoders.cnn
    import src.models.posteriors.diagonal
    import src.models.priors.standard_gaussian
    import src.models.likelihoods.gaussian_homo
    import src.models.losses.elbo

    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)

    assert hasattr(vae, "encoder")
    assert hasattr(vae, "posterior")
    assert hasattr(vae, "prior")
    assert hasattr(vae, "decoder")
    assert hasattr(vae, "likelihood")
    assert hasattr(vae, "loss_fn")

    x = torch.randn(2, 3, 32, 32)
    out = vae(x)
    assert out.reconstruction.shape == (2, 3, 32, 32)
