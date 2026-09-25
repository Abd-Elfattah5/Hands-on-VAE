"""Unit tests for analytical ELBO loss formulation and KL divergence."""

import torch


def test_analytical_kl_divergence():
    """Verify closed-form analytical KL divergence against known values."""
    from src.models.posteriors.diagonal import DiagonalGaussianDistribution
    from src.models.priors.standard_gaussian import StandardGaussianPrior

    prior = StandardGaussianPrior(latent_dim=2)

    # When q(z|x) matches prior N(0, I): mu = 0, log_var = 0 -> KL must be exactly 0
    mu_zero = torch.zeros(2, 2)
    log_var_zero = torch.zeros(2, 2)
    post_zero = DiagonalGaussianDistribution(mu=mu_zero, log_var=log_var_zero)

    kl = prior.kl_divergence(post_zero)
    assert torch.allclose(kl, torch.zeros(2), atol=1e-6)

    # Hand-calculated case: mu = [1, 0], var = [1, 1] (log_var = [0, 0])
    # KL = -0.5 * [(1 + 0 - 1^2 - 1) + (1 + 0 - 0^2 - 1)] = -0.5 * [-1 + 0] = 0.5
    mu_test = torch.tensor([[1.0, 0.0]])
    log_var_test = torch.tensor([[0.0, 0.0]])
    post_test = DiagonalGaussianDistribution(mu=mu_test, log_var=log_var_test)

    kl_test = prior.kl_divergence(post_test)
    assert torch.allclose(kl_test, torch.tensor([0.5]), atol=1e-5)


def test_elbo_loss_execution_and_gradients():
    """Verify ELBO loss computes finite values and propagates gradients."""
    from src.models.likelihoods.gaussian_homo import HomoscedasticGaussianLikelihood
    from src.models.losses.elbo import ELBOLoss
    from src.models.posteriors.diagonal import DiagonalGaussianDistribution
    from src.models.priors.standard_gaussian import StandardGaussianPrior
    from src.models.types import VAEOutput

    likelihood = HomoscedasticGaussianLikelihood(fixed_sigma=1.0)
    loss_fn = ELBOLoss(likelihood=likelihood, beta=1.0)

    recon = torch.zeros(2, 3, 32, 32, requires_grad=True)
    target = torch.ones(2, 3, 32, 32)
    z = torch.randn(2, 32, requires_grad=True)
    post = DiagonalGaussianDistribution(
        mu=torch.zeros(2, 32, requires_grad=True),
        log_var=torch.zeros(2, 32, requires_grad=True),
    )
    prior = StandardGaussianPrior(latent_dim=32)

    vae_out = VAEOutput(
        reconstruction=recon,
        z=z,
        posterior=post,
        prior=prior,
    )

    loss_output = loss_fn(vae_out, target)

    assert torch.isfinite(loss_output.loss)
    assert loss_output.reconstruction_loss.item() > 0.0
    assert torch.isfinite(loss_output.kl_divergence)

    loss_output.loss.backward()
    assert recon.grad is not None
    assert torch.all(torch.isfinite(recon.grad))
