"""Variational posterior models."""

from src.models.posteriors.diagonal import (
    DiagonalGaussianDistribution,
    DiagonalGaussianPosterior,
)

__all__ = ["DiagonalGaussianPosterior", "DiagonalGaussianDistribution"]
