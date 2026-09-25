"""Unit tests for continuous Gaussian reparameterization and gradient flow."""

import torch


def test_reparameterization_gradient_flow():
    """Verify that gradients propagate cleanly through mu and log_var via the reparameterization trick."""
    from src.models.posteriors.diagonal import DiagonalGaussianPosterior

    posterior_module = DiagonalGaussianPosterior(in_features=64, latent_dim=32)
    features = torch.randn(4, 64, requires_grad=True)

    dist = posterior_module(features)
    z = dist.sample()

    assert z.shape == (4, 32)
    assert z.requires_grad

    # Backpropagate a dummy scalar loss
    loss = z.sum()
    loss.backward()

    assert features.grad is not None
    assert torch.all(torch.isfinite(features.grad))
    assert not torch.all(features.grad == 0.0)


def test_reparameterization_formula():
    """Verify z = mu + sigma * epsilon relationship under fixed epsilon."""
    from src.models.posteriors.diagonal import DiagonalGaussianPosterior, DiagonalGaussianDistribution

    mu = torch.tensor([[1.0, -2.0, 3.0]])
    log_var = torch.tensor([[0.0, 0.0, 0.0]])  # sigma = exp(0) = 1.0
    epsilon = torch.tensor([[0.5, -0.5, 1.0]])

    dist = DiagonalGaussianDistribution(mu=mu, log_var=log_var)
    z = dist.rsample(eps=epsilon)

    expected = mu + torch.exp(0.5 * log_var) * epsilon
    assert torch.allclose(z, expected, atol=1e-5)
