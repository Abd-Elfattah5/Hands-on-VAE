"""Unit tests for dimensional preservation and spatial shape contracts."""

import torch


def test_cnn_encoder_shapes():
    """Verify 3-stage CNN encoder spatial downsampling (32 -> 16 -> 8 -> 4)."""
    from src.models.encoders.cnn import CNNEncoder

    encoder = CNNEncoder(
        in_channels=3,
        channels=[32, 64, 128],
        image_size=32,
        num_groups=8,
        act_func="leaky_relu",
        negative_slope=0.2,
    )

    x = torch.randn(4, 3, 32, 32)
    features = encoder(x)

    assert encoder.out_features == 128 * 4 * 4  # 2048
    assert features.shape == (4, 2048)


def test_cnn_decoder_shapes_and_bounds():
    """Verify 3-stage transposed CNN decoder upsampling and tanh bounding in [-1, 1]."""
    from src.models.decoders.cnn import CNNDecoder

    decoder = CNNDecoder(
        latent_dim=32,
        channels=[128, 64, 32],
        out_channels=3,
        image_size=32,
        num_groups=8,
        act_func="leaky_relu",
        negative_slope=0.2,
        final_act="tanh",
    )

    z = torch.randn(4, 32)
    recon = decoder(z)

    assert recon.shape == (4, 3, 32, 32)
    assert recon.min() >= -1.0
    assert recon.max() <= 1.0


def test_full_vae_forward_shapes():
    """Verify end-to-end VAE forward pass shape contracts."""
    from src.configs.schema import load_config
    from src.models.registry import build_vae_from_config
    # Ensure components are registered
    import src.models.encoders.cnn
    import src.models.decoders.cnn
    import src.models.posteriors.diagonal
    import src.models.priors.standard_gaussian
    import src.models.likelihoods.gaussian_homo
    import src.models.losses.elbo

    cfg = load_config("configs/cifar10_baseline.yaml")
    vae = build_vae_from_config(cfg)

    x = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
    output = vae(x)

    assert output.reconstruction.shape == (2, 3, 32, 32)
    assert output.z.shape == (2, 32)
    assert output.reconstruction.min() >= -1.0
    assert output.reconstruction.max() <= 1.0
