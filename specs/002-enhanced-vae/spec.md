# Feature Specification: 002-enhanced-vae

**Feature Branch**: `002-enhanced-vae`  
**Created**: September 25, 2026  
**Status**: In Design  
**Input**: Enhanced VAE with heteroscedastic pixel-wise likelihood, beta-NLL loss stabilization, and expanded latent capacity ($d=128$) on CIFAR-10.

---

## User Scenarios & Testing

### User Story 1 - Train Enhanced VAE with Heteroscedastic Likelihood (Priority: P1) 🎯 MVP

As a generative modeling engineer, I want to train a VAE with a pixel-wise heteroscedastic Gaussian likelihood and $\beta$-NLL loss stabilization on CIFAR-10, so that the model can learn spatially adaptive uncertainty, mitigating the $L_2$ conditional mean smoothing trap and reducing image blurriness.

**Why this priority**: Directly addresses the root mathematical failure mode identified in the baseline report (`001-baseline-vae-report.md`). By allowing the decoder to predict pixel-level variance $\sigma_{\text{obs}}^2(x, y)$ while downweighting high-variance regions via $\beta$-NLL, the model is freed from the uniform MSE penalty that causes severe edge blurriness.

**Independent Test**:
```bash
vae verify --config configs/cifar10_enhanced.yaml
vae train --config configs/cifar10_enhanced.yaml --epochs 50 --batch-size 128
```
Verifies end-to-end forward/backward gradient flow through both mean and variance heads and convergence over 50 epochs.

**Acceptance Scenarios**:
1. **Given** CIFAR-10 training images normalized to $[-1, 1]$ and `configs/cifar10_enhanced.yaml`, **When** training is executed on the local NVIDIA Quadro T2000 GPU, **Then** the decoder emits both a bounded mean tensor $\mu \in [-1, 1]$ and a clamped log-variance map $\log \sigma_{\text{obs}}^2 \in [-10.0, 5.0]$.
2. **Given** predicted mean $\mu$ and variance map $\sigma_{\text{obs}}^2$, **When** the loss is computed, **Then** $\beta$-NLL weighting with $\beta = 0.5$ stabilizes optimization without variance explosion or vanishing gradients.
3. **Given** 50 epochs of training, **When** completion is reached, **Then** checkpoints (`best_checkpoint.pt`, `final_checkpoint.pt`), metrics (`metrics.json`), and loss curves (`loss_curve.png`) are persisted to `artifacts/runs/cifar10_enhanced/`.

---

### User Story 2 - Expanded Latent Representation Capacity ($d=128$) (Priority: P2)

As a generative vision researcher, I want to scale the latent space bottleneck from $d=32$ to $d=128$, so that the latent code can retain fine spatial details and complex semantic features without bottleneck saturation.

**Why this priority**: Natural photographic images on CIFAR-10 have high intrinsic visual complexity. Scaling the latent dimensionality to $d=128$ increases representation bandwidth by $4\times$ while maintaining a peak GPU VRAM footprint under $250\text{ MB}$.

**Independent Test**:
`vae evaluate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt` reports active latent units and lower reconstruction error than the $d=32$ baseline.

**Acceptance Scenarios**:
1. **Given** $d=128$ latent configuration, **When** the encoder forward pass is executed, **Then** it projects 2048-dimensional spatial features into 128-dimensional posterior parameters $\mu, \log \sigma^2$.
2. **Given** test partition evaluation, **When** empirical variance per latent unit is measured, **Then** active units $A_z > 32$, demonstrating expanded informational capacity.

---

### User Story 3 - Comparative Quantitative & Qualitative Benchmarking (Priority: P3)

As an ML practitioner, I want to evaluate the enhanced VAE against the baseline on CIFAR-10 using Fréchet Inception Distance (FID), Inception Score (IS), test ELBO, reconstruction PSNR, and $1024 \times 1024$ visual plates, so that the empirical effect of heteroscedastic $\beta$-NLL and expanded capacity can be quantified.

**Why this priority**: Fulfills the comparative benchmarking requirements of `GenCV003`, providing objective numerical evidence of visual sharpness improvements.

**Independent Test**:
```bash
vae benchmark --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --num-samples 5000
vae generate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --num-samples 64 --upscale 4
vae interpolate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --synthetic --steps 8 --upscale 4
```

**Acceptance Scenarios**:
1. **Given** the 50-epoch enhanced checkpoint, **When** `vae benchmark` runs over 5,000 samples, **Then** the resulting Fréchet Inception Distance improves over the baseline FID of $169.02$.
2. **Given** qualitative sample grids and synthetic 2D prior interpolations, **When** visual artifacts are inspected, **Then** synthesized images display sharper edge contours and reduced pixel-averaging blur.

---

## Edge Cases

- **Variance Cheating / Collapse**: If unconstrained, a heteroscedastic network can predict $\sigma_{\text{obs}}^2 \to \infty$ on difficult edge pixels to artificially drive the squared error penalty $(x - \mu)^2 / \sigma^2 \to 0$. Handled by clamping $\log \sigma_{\text{obs}}^2 \in [-10.0, 5.0]$ and applying $\beta$-NLL downweighting with $(\sigma_{\text{obs}}^{2\beta})_{\text{detached}}$.
- **GPU VRAM Ceiling**: Full $3072 \times 3072$ pixel covariance is mathematically prohibited due to $4.6\text{ GB}$ allocation exceeding GPU limits. Pixel-wise heteroscedastic diagonal variance map ($3 \times 32 \times 32$) is strictly enforced, bounding peak VRAM to $\approx 240\text{ MB}$.
- **Backward Compatibility**: All existing baseline CLI commands and checkpoint restores for `001-create-vae` must continue to function without regression.

---

## Requirements

### Functional Requirements

- **FR-001**: System MUST implement a heteroscedastic CNN decoder (`HeteroscedasticCNNDecoder`) registered as `"cnn_hetero"` predicting both continuous observation mean $\mu \in \mathbb{R}^{B \times 3 \times 32 \times 32}$ bounded by `tanh` in $[-1, 1]$ and log-variance map $\log \sigma_{\text{obs}}^2 \in \mathbb{R}^{B \times 3 \times 32 \times 32}$ clamped to $[-10.0, 5.0]$.
- **FR-002**: System MUST implement a heteroscedastic Gaussian likelihood (`HeteroscedasticGaussianLikelihood`) registered as `"gaussian_hetero"` computing $\beta$-NLL loss:
  $$\mathcal{L}_{\beta\text{-NLL}} = \frac{1}{2} \sum_{i=1}^D \left[ \frac{(x_i - \mu_i)^2}{\sigma_i^2} + \log \sigma_i^2 \right] \cdot (\sigma_i^{2\beta})_{\text{detached}}$$
  with configurable parameter $\beta \in [0.0, 1.0]$ (default $\beta = 0.5$).
- **FR-003**: System MUST support configurable latent dimensionality $d = 128$ in declarative YAML configurations, correctly matching the linear projection dimensions in the encoder, posterior, and decoder.
- **FR-004**: System MUST provide a declarative configuration `configs/cifar10_enhanced.yaml` specifying `latent_dim: 128`, `decoder.name: "cnn_hetero"`, and `likelihood.name: "gaussian_hetero"` with 50 training epochs.
- **FR-005**: System MUST compute and log decomposed training diagnostics: total loss, reconstruction loss, analytical KL divergence, raw pixel MSE, and mean predicted observation variance $\bar{\sigma}_{\text{obs}}^2$.
- **FR-006**: System MUST persist the variance map in `VAEOutput.extra["log_var_map"]` to allow visual inspection and uncertainty mapping.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: Enhanced VAE with heteroscedastic likelihood and $d=128$ trains on NVIDIA Quadro T2000 GPU under 1.5 GB VRAM peak memory without out-of-memory errors.
- **SC-002**: Quantitative benchmarking on 5,000 CIFAR-10 test samples achieves a Fréchet Inception Distance (FID) strictly lower than the baseline FID of **$169.02$**.
- **SC-003**: Reconstruction fidelity on CIFAR-10 test split improves over the baseline reconstruction MSE of **$0.0578$** (PSNR $> 18.1\text{ dB}$).
- **SC-004**: Active latent dimensions count $A_z > 32$, confirming that the expanded capacity is actively utilized across the 128 latent axes.
- **SC-005**: All existing 35 unit tests pass without regression, and new unit tests for the heteroscedastic decoder and $\beta$-NLL likelihood pass with 100% success rate.
