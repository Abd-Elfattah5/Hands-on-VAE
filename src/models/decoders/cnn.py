"""Convolutional neural network decoder backbone."""

from typing import Sequence
import torch
import torch.nn as nn

from src.models.base import BaseDecoder
from src.models.registry import register_decoder


@register_decoder("cnn")
class CNNDecoder(BaseDecoder):
    """3-stage transposed convolutional decoder with GroupNorm and tanh bounding."""

    def __init__(
        self,
        latent_dim: int = 32,
        channels: Sequence[int] = (128, 64, 32),
        out_channels: int = 3,
        image_size: int = 32,
        num_groups: int = 8,
        act_func: str = "leaky_relu",
        negative_slope: float = 0.2,
        final_act: str = "tanh",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._channels = list(channels)
        self.out_channels = out_channels
        self.image_size = image_size
        self.num_groups = num_groups

        num_upsamples = len(self._channels)
        self._init_spatial_dim = image_size // (2 ** num_upsamples)
        self._init_channels = self._channels[0]
        self._fc_out_dim = self._init_channels * (self._init_spatial_dim ** 2)

        # Projection from latent space to initial spatial feature volume
        self.fc_in = nn.Linear(latent_dim, self._fc_out_dim)

        # Build transposed convolution upsampling stages
        deconv_layers: list[nn.Module] = []
        cur_in = self._init_channels

        # We upsample through stages: 4 -> 8 -> 16 -> 32
        for i, out_ch in enumerate(self._channels[1:] + [self._channels[-1]]):
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

        # Output head projecting to RGB image channels with tanh activation
        head_layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=cur_in,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        ]
        if final_act == "tanh":
            head_layers.append(nn.Tanh())
        elif final_act == "sigmoid":
            head_layers.append(nn.Sigmoid())
        elif final_act is not None and final_act != "none":
            raise ValueError(f"Unsupported final activation: {final_act}")

        self.head = nn.Sequential(*head_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into continuous observation parameters [-1, 1].

        Args:
            z: Latent tensor of shape [B, latent_dim].

        Returns:
            Reconstruction tensor of shape [B, out_channels, H, W].
        """
        x_init = self.fc_in(z)
        x_spatial = x_init.view(-1, self._init_channels, self._init_spatial_dim, self._init_spatial_dim)
        features = self.deconv_net(x_spatial)
        return self.head(features)
