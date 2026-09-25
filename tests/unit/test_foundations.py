"""Unit tests for Phase 2 foundational components (types, base, registry, schema, data)."""

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.configs.schema import (
    ConfigError,
    load_config,
    validate_config,
    validate_model_config,
)
from src.data.cifar10 import (
    get_cifar10_transforms,
    unnormalize,
    vae_collate_fn,
)
from src.models.base import (
    BaseDecoder,
    BaseEncoder,
    BaseLikelihood,
    BaseLoss,
    BasePosterior,
    BasePosteriorDistribution,
    BasePrior,
    BasePriorDistribution,
)
from src.models.registry import (
    DECODER_REGISTRY,
    ENCODER_REGISTRY,
    LIKELIHOOD_REGISTRY,
    LOSS_REGISTRY,
    POSTERIOR_REGISTRY,
    PRIOR_REGISTRY,
    get_decoder,
    get_encoder,
    get_likelihood,
    get_loss,
    get_posterior,
    get_prior,
    register_decoder,
    register_encoder,
    register_likelihood,
    register_loss,
    register_posterior,
    register_prior,
)
from src.models.types import LossOutput, VAEOutput


# -------------------------------------------------------------------------
# T005: Types tests
# -------------------------------------------------------------------------
def test_vae_output_creation():
    recon = torch.zeros(2, 3, 32, 32)
    z = torch.zeros(2, 32)
    out = VAEOutput(
        reconstruction=recon,
        z=z,
        posterior=None,
        prior=None,
        extra={"aux": 1},
    )
    assert out.reconstruction.shape == (2, 3, 32, 32)
    assert out.z.shape == (2, 32)
    assert out.extra["aux"] == 1


def test_loss_output_creation():
    loss = torch.tensor(1.5, requires_grad=True)
    recon_loss = torch.tensor(1.0)
    kl_div = torch.tensor(0.5)
    metrics = {"recon": 1.0, "kl": 0.5}

    out = LossOutput(
        loss=loss,
        reconstruction_loss=recon_loss,
        kl_divergence=kl_div,
        metrics=metrics,
    )
    assert out.loss.requires_grad
    assert out.reconstruction_loss.item() == 1.0
    assert out.kl_divergence.item() == 0.5
    assert out.metrics["recon"] == 1.0


# -------------------------------------------------------------------------
# T006: Base classes tests
# -------------------------------------------------------------------------
def test_abstract_base_classes_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseEncoder()
    with pytest.raises(TypeError):
        BaseDecoder()
    with pytest.raises(TypeError):
        BasePosterior()
    with pytest.raises(TypeError):
        BasePrior()
    with pytest.raises(TypeError):
        BaseLikelihood()
    with pytest.raises(TypeError):
        BaseLoss()
    with pytest.raises(TypeError):
        BasePosteriorDistribution()
    with pytest.raises(TypeError):
        BasePriorDistribution()


def test_concrete_subclass_instantiation():
    class DummyEncoder(BaseEncoder):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flatten(1)

        @property
        def out_features(self) -> int:
            return 128

    enc = DummyEncoder()
    assert enc.out_features == 128
    dummy_x = torch.zeros(2, 4, 4)
    assert enc(dummy_x).shape == (2, 16)


# -------------------------------------------------------------------------
# T007: Registry tests
# -------------------------------------------------------------------------
def test_registries_and_decorators():
    # Test valid registration
    @register_encoder("dummy_test_enc")
    class DummyEnc(BaseEncoder):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

        @property
        def out_features(self) -> int:
            return 32

    assert "dummy_test_enc" in ENCODER_REGISTRY
    assert get_encoder("dummy_test_enc") is DummyEnc

    # Test invalid registration (not subclass of BaseEncoder)
    with pytest.raises(TypeError):
        @register_encoder("invalid_enc")
        class NotAnEncoder:
            pass

    # Test unknown component retrieval
    with pytest.raises(KeyError) as exc_info:
        get_encoder("nonexistent_encoder")
    assert "nonexistent_encoder" in str(exc_info.value)


# -------------------------------------------------------------------------
# T008: Schema tests
# -------------------------------------------------------------------------
def test_validate_cifar10_baseline_config():
    cfg = load_config("configs/cifar10_baseline.yaml")
    exp_cfg = validate_config(cfg)
    assert exp_cfg.experiment.name == "cifar10_baseline"
    assert exp_cfg.model.latent_dim == 32
    assert exp_cfg.data.batch_size == 128
    assert exp_cfg.training.epochs == 50


def test_validate_mnist_baseline_config():
    cfg = load_config("configs/mnist_baseline.yaml")
    exp_cfg = validate_config(cfg)
    assert exp_cfg.experiment.name == "mnist_baseline"
    assert exp_cfg.model.in_channels == 1
    assert exp_cfg.model.latent_dim == 32
    assert exp_cfg.data.dataset == "mnist"


def test_schema_dimensional_validation():
    cfg = load_config("configs/cifar10_baseline.yaml")

    # Incompatible image size and downsample depth
    bad_cfg = dict(cfg)
    bad_cfg["model"] = dict(cfg["model"])
    bad_cfg["model"]["image_size"] = 15  # Not divisible by 2^3=8
    with pytest.raises(ConfigError) as exc_info:
        validate_config(bad_cfg)
    assert "divisible" in str(exc_info.value)

    # Incompatible group normalization
    bad_cfg2 = dict(cfg)
    bad_cfg2["model"] = dict(cfg["model"])
    bad_cfg2["model"]["encoder"] = dict(cfg["model"]["encoder"])
    bad_cfg2["model"]["encoder"]["num_groups"] = 7  # 32 is not divisible by 7
    with pytest.raises(ConfigError) as exc_info:
        validate_config(bad_cfg2)
    assert "num_groups" in str(exc_info.value)

    # Clamp bounds inversion
    bad_cfg3 = dict(cfg)
    bad_cfg3["model"] = dict(cfg["model"])
    bad_cfg3["model"]["posterior"] = dict(cfg["model"]["posterior"])
    bad_cfg3["model"]["posterior"]["clamp_min"] = 3.0
    bad_cfg3["model"]["posterior"]["clamp_max"] = 1.0
    with pytest.raises(ConfigError):
        validate_config(bad_cfg3)


# -------------------------------------------------------------------------
# T009: Data tests
# -------------------------------------------------------------------------
def test_cifar10_transforms():
    transforms = get_cifar10_transforms()
    # Test on a dummy PIL image (all zeros and all 255)
    img_black = Image.new("RGB", (32, 32), (0, 0, 0))
    img_white = Image.new("RGB", (32, 32), (255, 255, 255))

    t_black = transforms(img_black)
    t_white = transforms(img_white)

    # Pixel 0 normalized: (0 - 0.5) / 0.5 = -1.0
    assert torch.allclose(t_black, torch.full_like(t_black, -1.0), atol=1e-3)
    # Pixel 255 normalized: (1.0 - 0.5) / 0.5 = 1.0
    assert torch.allclose(t_white, torch.full_like(t_white, 1.0), atol=1e-3)

    # Test unnormalize
    unnorm_black = unnormalize(t_black)
    unnorm_white = unnormalize(t_white)
    assert torch.allclose(unnorm_black, torch.zeros_like(unnorm_black), atol=1e-3)
    assert torch.allclose(unnorm_white, torch.ones_like(unnorm_white), atol=1e-3)


def test_vae_collate_fn():
    # Valid batch within [-1, 1]
    valid_batch = [
        (torch.tensor([[-0.5, 0.5]]), torch.tensor(0)),
        (torch.tensor([[0.0, 1.0]]), torch.tensor(1)),
    ]
    imgs, targets = vae_collate_fn(valid_batch)
    assert imgs.shape[0] == 2
    assert targets.shape[0] == 2

    # Out of bounds batch
    invalid_batch = [
        (torch.tensor([[2.5, 0.5]]), torch.tensor(0)),
    ]
    with pytest.raises(ValueError) as exc_info:
        vae_collate_fn(invalid_batch)
    assert "outside [-1, 1] bounds" in str(exc_info.value)


def test_mnist_transforms():
    from src.data.mnist import get_mnist_transforms
    transforms = get_mnist_transforms()

    img_black = Image.new("L", (28, 28), 0)
    t = transforms(img_black)

    # 28x28 padded with 2 on all sides -> 32x32
    assert t.shape == (1, 32, 32)
    # 0 normalized with mean=0.5, std=0.5 -> -1.0
    assert torch.allclose(t, torch.full_like(t, -1.0), atol=1e-3)
