"""Decoder architectures."""

from src.models.decoders.cnn import CNNDecoder
from src.models.decoders.cnn_hetero import HeteroscedasticCNNDecoder

__all__ = ["CNNDecoder", "HeteroscedasticCNNDecoder"]
