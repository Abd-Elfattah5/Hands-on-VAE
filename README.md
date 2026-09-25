# Hands-on Variational Autoencoder (VAE) on CIFAR-10

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, from-scratch implementation and benchmarking pipeline for Variational Autoencoders in PyTorch. Developed for the `GenCV003` deep generative modeling benchmark.

---

## Overview

This repository contains a ground-up implementation of a continuous **Variational Autoencoder (VAE)** benchmarked on **CIFAR-10** ($32 \times 32 \times 3$). Built strictly from foundational PyTorch tensor primitives, all encoder/decoder topologies, reparameterization mechanics, and analytical objectives avoid external generative wrappers.

### Key Features
* **From-Scratch Mathematical Modeling**: Analytical Evidence Lower Bound (ELBO) loss decomposition, continuous Gaussian reparameterization ($z = \mu + \sigma \odot \epsilon$), and closed-form analytical KL divergence.
* **Extensible Component Registries**: Pluggable factory architecture covering Encoders, Posteriors, Priors, Decoders, Likelihoods, and Losses (`@register_encoder`, etc.).
* **Deterministic Reproducibility**: Unified global seeding across Python, NumPy, and PyTorch (CPU/CUDA) with complete checkpoint provenance metadata.
* **Standardized Benchmarking Pipeline**: Quantitative evaluation computing test ELBO, reconstruction MSE, active latent units ($A_z = 32/32$), **Fréchet Inception Distance (FID)**, and **Inception Score (IS)**.
* **Rich Qualitative Diagnostics**: High-resolution canvas upscaling (Lanczos $4\times$, $1024 \times 1024$ grids), pure synthetic prior interpolation, 1D latent coordinate sweeps, 2D generative manifold surface traversals, and t-SNE / PCA latent space projections.

---

## Quantitative Benchmark Results (CIFAR-10, 50 Epochs)

| Metric | Baseline VAE (`001`) | Enhanced VAE (`002`) | Notes |
| :--- | :---: | :---: | :--- |
| **Observation Likelihood** | Homoscedastic (Fixed $\sigma^2=1.0$) | Heteroscedastic + $\beta$-NLL ($\beta=0.5$) | Pixel-wise adaptive uncertainty |
| **Latent Dimension ($d$)** | **32** | **128** | $4\times$ expanded bottleneck bandwidth |
| **Model Parameters** | **1.60 M** | **2.25 M** | $+0.65\text{M}$ parameters |
| **Reconstruction MSE (Test)** | **0.0578** | **0.0563** | Test partition pixel error (Improved) |
| **Gallery Test PSNR** | **18.1 dB** (MSE: 0.0156) | **18.3 dB** (MSE: 0.0147) | Evaluated on real test images (Improved) |
| **KL Divergence** | **33.82 nats** | **92.32 nats** | $\approx 0.72$ nats / dimension |
| **Active Latent Units ($A_z$)** | **32 / 32** | **128 / 128** | $100\%$ capacity utilization ($\text{Var}(\mu_j) > 0.01$) |
| **Fréchet Inception Distance (FID)** | **169.02** | **181.00** | Evaluated on 5,000 samples |
| **Inception Score (IS)** | **2.11 ± 0.03** | **1.68 ± 0.04** | Evaluated on 5,000 samples |
| **Peak GPU VRAM** | **1.82 GB** | **1.84 GB** | Local NVIDIA Quadro T2000 (Budget: $<2.5\text{ GB}$) |

Detailed mathematical derivations, training curves, and analysis are available in:
* [Baseline Technical Report (CIFAR-10 & MNIST)](docs/reports/001-baseline-vae-report.md)
* [Enhanced VAE Technical Report (Heteroscedastic $\beta$-NLL)](docs/reports/002-enhanced-vae-report.md)

---

## Project Structure

```text
├── configs/
│   └── cifar10_baseline.yaml       # Declarative experiment configuration
├── docs/
│   ├── adr/                         # Architecture Decision Records
│   └── reports/
│       └── 001-baseline-vae-report.md # Formal VAE Technical Report
├── specs/                           # Feature specs, task lists, and contracts
│   └── 001-create-vae/
├── src/
│   ├── cli/                         # Unified Typer CLI entrypoint
│   ├── configs/                     # Typed YAML schema and dimensional validator
│   ├── data/                        # CIFAR-10 transforms, bounds checking, dataloaders
│   ├── evaluation/                  # Evaluator, FID/IS metrics, visualizer, latent analysis
│   ├── models/                      # Encoders, decoders, posteriors, priors, losses
│   ├── training/                    # Trainer loop, scheduler, and checkpoint manager
│   └── utils/                       # Global seeding and structured logging
└── tests/
    └── unit/                        # 33 unit tests (100% passing)
```

---

## Installation & Setup

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Abd-Elfattah5/Hands-on-VAE.git
cd Hands-on-VAE

# Create and activate virtual environment
python3 -m venv VAE
source VAE/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Verify Architecture & Tensor Integrity

Before training, execute the pre-training integrity gate:

```bash
vae verify --config configs/cifar10_baseline.yaml
```

---

## CLI Reproduction Guide

The `vae` command-line interface provides complete operational workflows:

### 1. Train Baseline VAE (50 Epochs)

```bash
vae train --config configs/cifar10_baseline.yaml --epochs 50 --batch-size 128
```
*Emits `best_checkpoint.pt`, `final_checkpoint.pt`, `metrics.json`, and `loss_curve.png` to `artifacts/runs/cifar10_baseline/`.*

### 2. Evaluate Held-Out Test Split

```bash
vae evaluate --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt
```

### 3. Quantitative Benchmarking (FID & Inception Score)

```bash
vae benchmark --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt --num-samples 5000
```
*Calculates Fréchet Inception Distance and Inception Score over 5,000 real vs. synthetic samples and persists metrics to `artifacts/eval/benchmark_metrics.json`.*

### 4. Qualitative Latent Space Diagnostics (t-SNE, PCA, 2D Manifolds)

```bash
vae plot-latent --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt --num-samples 2500 --out-dir artifacts/eval
```
*Generates:*
* `artifacts/eval/latent_tsne.png`: 2D t-SNE scatter colored by CIFAR-10 class labels with prior overlay.
* `artifacts/eval/latent_pca.png`: 2D PCA projection showing principal variance components.
* `artifacts/eval/latent_manifold_2d.png`: Continuous 2D generative manifold meshgrid.
* `artifacts/eval/latent_traversal_1d.png`: 1D coordinate sweeps along individual latent axes.
* `artifacts/eval/reconstruction_gallery.png`: Side-by-side real vs. reconstructed image pairs with PSNR/MSE metrics.

### 5. Generate High-Resolution Prior Sample Grids

```bash
vae generate \
  --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt \
  --num-samples 64 \
  --out artifacts/samples/sample_grid_1024.png \
  --upscale 4 \
  --seed 42
```
*Outputs a crisp $1024 \times 1024$ canvas of synthetic samples drawn from $\mathcal{N}(0, I)$.*

### 6. Pure Synthetic 2D Prior Interpolation

```bash
vae interpolate \
  --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt \
  --synthetic \
  --steps 8 \
  --out artifacts/interpolations/synthetic_grid_2d.png \
  --upscale 4
```

### 7. Run Test Suite

```bash
pytest tests/ -v
```
*Executes all 33 unit tests verifying shape preservation, gradient propagation through reparameterization, analytical KL formulas, FID, and Inception Score calculations.*

---

## License

This project is licensed under the MIT License.
