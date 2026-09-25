"""Unit tests for heteroscedastic CNN decoder and beta-NLL Gaussian likelihood."""

import pytest
import torch


def test_heteroscedastic_decoder_shapes_and_bounds():
    """Verify dual output shapes and clamp ranges for HeteroscedasticCNNDecoder."""
    from src.models.decoders.cnn_hetero import HeteroscedasticCNNDecoder

    decoder = HeteroscedasticCNNDecoder(
        latent_dim=128,
        channels=[128, 64, 32],
        out_channels=3,
        image_size=32,
        num_groups=8,
        act_func="leaky_relu",
        negative_slope=0.2,
        final_act="tanh",
        clamp_min=-10.0,
        clamp_max=5.0,
    )

    z = torch.randn(4, 128)
    mu, log_var = decoder(z)

    assert mu.shape == (4, 3, 32, 32)
    assert log_var.shape == (4, 3, 32, 32)
    assert mu.min() >= -1.0
    assert mu.max() <= 1.0
    assert log_var.min() >= -10.0
    assert log_var.max() <= 5.0


def test_beta_nll_likelihood_calculation_and_gradients():
    """Verify beta-NLL loss formula and gradient propagation."""
    from src.models.likelihoods.gaussian_hetero import HeteroscedasticGaussianLikelihood

    likelihood = HeteroscedasticGaussianLikelihood(beta_nll=0.5)

    mu = torch.zeros(2, 3, 32, 32, requires_grad=True)
    log_var = torch.zeros(2, 3, 32, 32, requires_grad=True)  # var = exp(0) = 1.0
    target = torch.ones(2, 3, 32, 32)

    extra = {"log_var_map": log_var}
    nll = likelihood.negative_log_likelihood(mu, target, extra=extra)

    assert nll.shape == (2,)
    assert torch.all(torch.isfinite(nll))
    assert torch.all(nll > 0.0)

    # When var=1, beta_weight=(1)^0.5 = 1, loss = 0.5 * (1^2 / 1 + 0) * 3072 = 1536.0
    expected_sample_loss = 0.5 * 3 * 32 * 32
    assert pytest.approx(nll[0].item(), rel=1e-3) == expected_sample_loss

    loss = nll.mean()
    loss.backward()

    assert mu.grad is not None
    assert log_var.grad is not None
    assert torch.all(torch.isfinite(mu.grad))
    assert torch.all(torch.isfinite(log_var.grad))


def test_latent_dim_128_forward_and_loss():
    """Verify 128-dimensional latent space forward pass and loss computation."""
    from src.configs.schema import load_config
    from src.models.registry import build_vae_from_config

    cfg = load_config("configs/cifar10_enhanced.yaml")
    vae = build_vae_from_config(cfg)

    x = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
    output = vae(x)

    assert output.z.shape == (2, 128)
    assert output.reconstruction.shape == (2, 3, 32, 32)
    assert "log_var_map" in output.extra
    assert output.extra["log_var_map"].shape == (2, 3, 32, 32)

    loss_output = vae.compute_loss(output, x)
    assert torch.isfinite(loss_output.loss)
    assert loss_output.loss.requires_grad
