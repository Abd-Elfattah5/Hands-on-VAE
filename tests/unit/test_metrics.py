"""Unit tests for quantitative benchmarking metrics (Fréchet distance, Inception Score)."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    calculate_frechet_distance,
    calculate_inception_score,
)


def test_frechet_distance_identical_distributions():
    """Verify Fréchet distance is zero when distributions are identical."""
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([
        [1.0, 0.1, 0.0],
        [0.1, 2.0, 0.2],
        [0.0, 0.2, 1.5],
    ])

    fd = calculate_frechet_distance(mu, sigma, mu, sigma)
    assert pytest.approx(fd, abs=1e-5) == 0.0


def test_frechet_distance_known_shift():
    """Verify Fréchet distance equals squared Euclidean distance when covariances are identical."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([3.0, 4.0])
    sigma = np.eye(2)

    # Expected: ||mu1 - mu2||^2 + Tr(sigma + sigma - 2 * sigma) = 3^2 + 4^2 + 0 = 25.0
    fd = calculate_frechet_distance(mu1, sigma, mu2, sigma)
    assert pytest.approx(fd, rel=1e-4) == 25.0


def test_inception_score_uniform_predictions():
    """Verify Inception Score equals 1.0 when predictions are uniform across classes."""
    # 100 samples, 10 classes, uniform probabilities -> KL(p(y|x) || p(y)) = 0 -> exp(0) = 1.0
    probs = np.full((100, 10), 0.1)
    is_mean, is_std = calculate_inception_score(probs, splits=5)

    assert pytest.approx(is_mean, rel=1e-4) == 1.0
    assert pytest.approx(is_std, abs=1e-5) == 0.0


def test_inception_score_diverse_confident_predictions():
    """Verify Inception Score is high when predictions are confident and uniformly distributed across classes."""
    # Each sample predicts one class with probability 1.0, and all 10 classes are equally represented
    num_samples = 100
    probs = np.zeros((num_samples, 10))
    for i in range(num_samples):
        probs[i, i % 10] = 1.0

    is_mean, is_std = calculate_inception_score(probs, splits=5)
    # Theoretical IS for 10 distinct uniform classes is exp(log(10)) = 10.0
    assert pytest.approx(is_mean, rel=1e-2) == 10.0
