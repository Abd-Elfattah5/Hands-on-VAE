# Quickstart Validation Guide: Standalone VAE

**Feature**: `001-create-vae`
**Status**: Ready for Implementation

This guide provides end-to-end runnable scenarios to validate the from-scratch VAE implementation on CIFAR-10.

---

## 1. Prerequisites & Environment Setup

Ensure the virtual environment is activated and dependencies installed:

```bash
# Activate virtual environment
source VAE/bin/activate

# Install package in editable development mode with test dependencies
pip install -e ".[dev]"
```

Verify GPU availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Validation Scenario 1: Architecture Integrity Verification

Before training, run the automated tensor and gradient integrity check:

```bash
vae verify --config configs/cifar10_baseline.yaml
```

**Expected Outcome**:
- Forward pass completes for batch size 2.
- Backward pass produces non-zero gradients on all encoder, decoder, and posterior weights.
- Loss computation produces finite scalar without NaN/Inf.
- Console outputs: `[SUCCESS] Tensor dimensions and gradient flow verified.`

---

## 3. Validation Scenario 2: Quick Smoke Training Run

Execute a 2-epoch smoke test to verify dataset download, dataloader batching, forward/backward execution, metrics logging, and checkpoint saving:

```bash
vae train --config configs/cifar10_baseline.yaml --epochs 2 --batch-size 128
```

**Expected Outcome**:
- CIFAR-10 downloads automatically into `data/` if not present.
- Training progress bar displays loss per batch.
- Validation loop executes after epoch 1 and 2.
- Checkpoints emitted:
  - `artifacts/runs/cifar10_baseline/best_checkpoint.pt`
  - `artifacts/runs/cifar10_baseline/final_checkpoint.pt`
  - `artifacts/runs/cifar10_baseline/metrics.json`
  - `artifacts/runs/cifar10_baseline/loss_curve.png`

---

## 4. Validation Scenario 3: Test Split Evaluation

Evaluate the saved checkpoint on the held-out test split:

```bash
vae evaluate --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt
```

**Expected Outcome**:
- Loads model weights and configuration schema cleanly.
- Outputs quantitative test ELBO, reconstruction loss, and KL divergence.
- Emits `eval_metrics.json`.

---

## 5. Validation Scenario 4: Prior Sampling & Visual Synthesis

Generate synthetic CIFAR-10 samples by sampling latent vectors from $\mathcal{N}(0, I)$:

```bash
vae generate \
  --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt \
  --num-samples 64 \
  --out artifacts/samples/cifar10_samples.png \
  --seed 42
```

**Expected Outcome**:
- Outputs a formatted $8 \times 8$ image grid to `artifacts/samples/cifar10_samples.png`.
- Re-running with `--seed 42` produces an identical image grid bit-for-bit.

---

## 6. Validation Scenario 5: Latent Space Interpolation

Generate a smooth transition strip between two CIFAR-10 test images:

```bash
vae interpolate \
  --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt \
  --img1 data/samples/test_img_0.png \
  --img2 data/samples/test_img_1.png \
  --steps 10 \
  --out artifacts/interpolations/walk.png
```

**Expected Outcome**:
- Encodes both images to $z_1, z_2 \in \mathbb{R}^{32}$.
- Computes 10 interpolated latents and decodes each.
- Emits an ordered horizontal strip showing continuous semantic transformation.

---

## 7. Validation Scenario 6: Automated Unit Test Suite

Execute the full suite of unit tests:

```bash
pytest tests/ -v
```

**Expected Outcome**:
- All unit tests pass:
  - `tests/unit/test_reparameterization.py`
  - `tests/unit/test_encoder_decoder_shapes.py`
  - `tests/unit/test_loss_formulations.py`
  - `tests/unit/test_registry.py`
  - `tests/unit/test_checkpoint_portability.py`
