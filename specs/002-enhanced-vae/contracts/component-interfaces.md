# Component Interface Contracts: 002-enhanced-vae

## 1. HeteroscedasticCNNDecoder Interface

```python
from src.models.base import BaseDecoder
from src.models.registry import register_decoder

@register_decoder("cnn_hetero")
class HeteroscedasticCNNDecoder(BaseDecoder):
    """3-stage transposed CNN decoder predicting both mean mu and pixel log-variance map."""

    def __init__(
        self,
        latent_dim: int = 128,
        channels: Sequence[int] = (128, 64, 32),
        out_channels: int = 3,
        image_size: int = 32,
        num_groups: int = 8,
        act_func: str = "leaky_relu",
        negative_slope: float = 0.2,
        final_act: str = "tanh",
        clamp_min: float = -10.0,
        clamp_max: float = 5.0,
    ) -> None:
        ...

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent z into (mu, log_var_map).

        Args:
            z: Latent tensor [B, latent_dim].

        Returns:
            Tuple of:
                - mu: Tensor [B, out_channels, H, W] in [-1, 1]
                - log_var: Tensor [B, out_channels, H, W] clamped to [clamp_min, clamp_max]
        """
        ...
```

---

## 2. HeteroscedasticGaussianLikelihood Interface

```python
from src.models.base import BaseLikelihood
from src.models.registry import register_likelihood

@register_likelihood("gaussian_hetero")
class HeteroscedasticGaussianLikelihood(BaseLikelihood):
    """Pixel-wise heteroscedastic Gaussian likelihood with beta-NLL stabilization."""

    def __init__(self, beta_nll: float = 0.5) -> None:
        super().__init__()
        self.beta_nll = beta_nll

    def negative_log_likelihood(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        extra: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute beta-NLL loss per batch element.

        Args:
            reconstruction: Predicted mean tensor [B, C, H, W].
            target: Ground truth target image [B, C, H, W].
            extra: Auxiliary dictionary containing 'log_var_map'.

        Returns:
            Tensor of shape [B] containing beta-NLL loss per sample.
        """
        ...

    def sample(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Return deterministic mean observation."""
        return reconstruction
```

---

## 3. Top-Level Composite VAE Adaptor

In `src/models/vae.py`:
When `self.decoder` returns a tuple `(recon, log_var_map)`:
```python
        decoder_output = self.decoder(z)
        if isinstance(decoder_output, tuple):
            reconstruction, log_var_map = decoder_output
            extra = {"log_var_map": log_var_map}
        else:
            reconstruction = decoder_output
            extra = {}

        return VAEOutput(
            reconstruction=reconstruction,
            z=z,
            posterior=posterior_dist,
            prior=self.prior,
            extra=extra,
        )
```
And in `self.decode(z)`:
```python
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        if isinstance(out, tuple):
            return out[0]
        return out
```
This enables backward compatibility with standard homoscedastic decoders while supporting heteroscedastic dual outputs.
