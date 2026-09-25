"""Heteroscedastic transposed convolutional neural network decoder."""

from typing import Sequence, Tuple
import torch
import torch.nn as nn

from src.models.base import BaseDecoder
from src.models.registry import register_decoder


@register_decoder("cnn_hetero")
class HeteroscedasticCNNDecoder(BaseDecoder):
    """3-stage transposed CNN decoder with dual heads predicting mean and pixel-wise log-variance."""

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
        super().__init__()
        self.latent_dim = latent_dim
        self._channels = list(channels)
        self.out_channels = out_channels
        self.image_size = image_size
        self.num_groups = num_groups
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        num_upsamples = len(self._channels)
        self._init_spatial_dim = image_size // (2 ** num_upsamples)
        self._init_channels = self._channels[0]
        self._fc_out_dim = self._init_channels * (self._init_spatial_dim ** 2)

        # Projection from latent space to initial spatial feature volume
        self.fc_in = nn.Linear(latent_dim, self._fc_out_dim)

        # Transposed convolution upsampling stages
        deconv_layers: list[nn.Module] = []
        cur_in = self._init_channels

        for out_ch in self._channels[1:] + [self._channels[-1]]:
            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=cur_in,
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            deconv_layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
            if act_func == "leaky_relu":
                deconv_layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            elif act_func == "relu":
                deconv_layers.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f"Unsupported activation function: {act_func}")
            cur_in = out_ch

        self.deconv_net = nn.Sequential(*deconv_layers)

        # Head 1: Continuous observation mean mu in [-1, 1]
        head_mu_layers: list[nn.Module] = [
            nn.Conv2d(cur_in, out_channels, kernel_size=3, stride=1, padding=1)
        ]
        if final_act == "tanh":
            head_mu_layers.append(nn.Tanh())
        elif final_act == "sigmoid":
            head_mu_layers.append(nn.Sigmoid())
        self.head_mu = nn.Sequential(*head_mu_layers)

        # Head 2: Pixel-wise log-variance map log sigma^2
        self.head_log_var = nn.Conv2d(cur_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector z into observation mean and pixel-wise log-variance map.

        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Tuple of (mu [B, out_channels, H, W], log_var [B, out_channels, H, W]).
        """
        x_init = self.fc_in(z)
        x_spatial = x_init.view(-1, self._init_channels, self._init_spatial_dim, self._init_spatial_dim)
        features = self.deconv_net(x_spatial)

        mu = self.head_mu(features)
        raw_log_var = self.head_log_var(features)
        log_var = torch.clamp(raw_log_var, min=self.clamp_min, max=self.clamp_max)

        return mu, log_var
