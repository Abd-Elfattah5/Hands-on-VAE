"""Quantitative generative evaluation metrics: Inception Score (IS) and Fréchet Inception Distance (FID)."""

from typing import Any, Optional, Tuple
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

from src.data.cifar10 import unnormalize
from src.evaluation.evaluator import evaluate_model
from src.models.vae import VAE


class InceptionFeatureExtractor(nn.Module):
    """Wraps pre-trained Inception-v3 to extract 2048-dim penultimate features and softmax probabilities."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        inception = inception_v3(weights=weights, transform_input=False).to(device)
        inception.eval()
        self.inception = inception
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process images [B, 3, 32, 32] in [0, 1] range.

        Args:
            x: Input images [B, 3, 32, 32] in [0, 1].

        Returns:
            Tuple of (features [B, 2048], probs [B, 1000]).
        """
        # Expand 1-channel images (e.g. MNIST) to 3 channels for Inception-v3
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize from 32x32 to 299x299 as expected by Inception-v3
        x_299 = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # Normalize to Inception's expected [-1, 1] input range
        x_norm = x_299 * 2.0 - 1.0

        # Pass through Inception layers up to pre-aux
        inc = self.inception
        x = inc.Conv2d_1a_3x3(x_norm)
        x = inc.Conv2d_2a_3x3(x)
        x = inc.Conv2d_2b_3x3(x)
        x = inc.maxpool1(x)
        x = inc.Conv2d_3b_1x1(x)
        x = inc.Conv2d_4a_3x3(x)
        x = inc.maxpool2(x)
        x = inc.Mixed_5b(x)
        x = inc.Mixed_5c(x)
        x = inc.Mixed_5d(x)
        x = inc.Mixed_6a(x)
        x = inc.Mixed_6b(x)
        x = inc.Mixed_6c(x)
        x = inc.Mixed_6d(x)
        x = inc.Mixed_6e(x)
        x = inc.Mixed_7a(x)
        x = inc.Mixed_7b(x)
        x = inc.Mixed_7c(x)
        x = inc.avgpool(x)
        x = inc.dropout(x)
        features = torch.flatten(x, 1)  # [B, 2048]

        logits = inc.fc(features)
        probs = F.softmax(logits, dim=1)  # [B, 1000]

        return features, probs


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute the Fréchet distance between two multivariate Gaussians N(mu1, sigma1) and N(mu2, sigma2).

    Formula:
        d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary error correction
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_inception_score(
    probs: np.ndarray,
    splits: int = 10,
) -> Tuple[float, float]:
    """Calculate Inception Score (IS) from class probability predictions.

    Formula:
        IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )
    """
    num_samples = probs.shape[0]
    split_size = num_samples // splits
    scores: list[float] = []

    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        p_y = np.expand_dims(np.mean(part, axis=0), 0)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        kl_div = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl_div))

    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def compute_fid_and_is(
    model: VAE,
    real_loader: DataLoader,
    num_samples: int = 5000,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Extract Inception features for real images and model-generated samples to compute FID and IS.

    Args:
        model: Trained VAE model.
        real_loader: DataLoader containing real CIFAR-10 test images.
        num_samples: Total number of samples for benchmarking.
        batch_size: Batch size for feature extraction.
        device: Target execution device.

    Returns:
        Dictionary with 'fid', 'inception_score_mean', and 'inception_score_std'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = InceptionFeatureExtractor(device=device)

    # 1. Extract Real Features
    real_features_list: list[np.ndarray] = []
    collected_real = 0

    pbar = tqdm(real_loader, desc="Extracting Real Features", leave=False)
    for batch in pbar:
        images = batch[0].to(device)
        images_01 = unnormalize(images)
        feats, _ = extractor(images_01)
        real_features_list.append(feats.cpu().numpy())
        collected_real += images.size(0)
        if collected_real >= num_samples:
            break

    real_features = np.concatenate(real_features_list, axis=0)[:num_samples]

    # 2. Extract Synthetic Features & Probabilities
    model.eval()
    fake_features_list: list[np.ndarray] = []
    fake_probs_list: list[np.ndarray] = []
    collected_fake = 0

    num_batches = int(np.ceil(num_samples / batch_size))
    for _ in tqdm(range(num_batches), desc="Synthesizing & Extracting VAE Samples", leave=False):
        cur_batch = min(batch_size, num_samples - collected_fake)
        if cur_batch <= 0:
            break
        samples = model.sample(num_samples=cur_batch, device=device)
        samples_01 = unnormalize(samples)
        feats, probs = extractor(samples_01)

        fake_features_list.append(feats.cpu().numpy())
        fake_probs_list.append(probs.cpu().numpy())
        collected_fake += cur_batch

    fake_features = np.concatenate(fake_features_list, axis=0)[:num_samples]
    fake_probs = np.concatenate(fake_probs_list, axis=0)[:num_samples]

    # 3. Compute Metrics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    is_mean, is_std = calculate_inception_score(fake_probs)

    return {
        "fid": round(fid_score, 2),
        "inception_score_mean": round(is_mean, 2),
        "inception_score_std": round(is_std, 2),
        "benchmark_samples": num_samples,
    }
