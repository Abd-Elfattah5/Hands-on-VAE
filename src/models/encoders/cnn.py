"""Convolutional neural network encoder backbone."""

from typing import Sequence
import torch
import torch.nn as nn

from src.models.base import BaseEncoder
from src.models.registry import register_encoder


@register_encoder("cnn")
class CNNEncoder(BaseEncoder):
    """3-stage convolutional encoder with GroupNorm and LeakyReLU."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        image_size: int = 32,
        num_groups: int = 8,
        act_func: str = "leaky_relu",
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._channels = list(channels)
        self._image_size = image_size
        self._num_groups = num_groups

        layers: list[nn.Module] = []
        cur_in = in_channels

        for out_ch in self._channels:
            layers.append(
                nn.Conv2d(
                    in_channels=cur_in,
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
            if act_func == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            elif act_func == "relu":
                layers.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f"Unsupported activation function: {act_func}")
            cur_in = out_ch

        self.net = nn.Sequential(*layers)

        # Compute output feature dimension after downsampling
        num_downsamples = len(self._channels)
        self._spatial_dim = image_size // (2 ** num_downsamples)
        self._out_features = self._channels[-1] * (self._spatial_dim ** 2)

    @property
    def out_features(self) -> int:
        """Flattened spatial feature vector dimension."""
        return self._out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input observation into flattened spatial feature vector.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Flattened feature tensor of shape [B, out_features].
        """
        features = self.net(x)
        return torch.flatten(features, start_dim=1)
