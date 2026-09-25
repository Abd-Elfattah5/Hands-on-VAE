# Quickstart & Validation Guide: 002-enhanced-vae

## Prerequisites
- Virtual environment activated (`source VAE/bin/activate`).
- Package installed (`pip install -e .`).
- CIFAR-10 data cached in `data/cifar-10-batches-py`.

---

## 1. Architecture & Tensor Gradient Verification

```bash
vae verify --config configs/cifar10_enhanced.yaml
```
**Expected Outcome**: Non-zero gradients verified across all layers, including mean and variance heads; exit code 0.

---

## 2. Train Enhanced VAE (50 Epochs)

```bash
vae train --config configs/cifar10_enhanced.yaml --epochs 50 --batch-size 128
```
**Expected Outcome**: 50 epochs completed, loss trajectories saved to `artifacts/runs/cifar10_enhanced/loss_curve.png`, and `best_checkpoint.pt` saved.

---

## 3. Held-Out Test Evaluation

```bash
vae evaluate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt
```
**Expected Outcome**: Outputs test ELBO, reconstruction MSE, and active latent units ($A_z > 32$).

---

## 4. Quantitative Benchmarking (FID & Inception Score)

```bash
vae benchmark --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --num-samples 5000 --out artifacts/eval_enhanced/benchmark_metrics.json
```
**Expected Outcome**: Fréchet Inception Distance computed over 5,000 samples improves over baseline FID of $169.02$.

---

## 5. Qualitative Visual Diagnostics

```bash
# 1024x1024 sample grid
vae generate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --num-samples 64 --out artifacts/samples/enhanced_sample_grid_1024.png --upscale 4

# Synthetic 2D prior manifold interpolation
vae interpolate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --synthetic --steps 8 --out artifacts/interpolations/enhanced_synthetic_grid_2d.png --upscale 4

# Latent space diagnostics
vae plot-latent --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --num-samples 2500 --out-dir artifacts/eval_enhanced
```
