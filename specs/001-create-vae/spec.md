# Feature Specification: Standalone Variational Autoencoder (VAE)

**Feature Branch**: `001-create-vae`

**Created**: 2026-09-23

**Status**: Ready for Planning

**Input**: User description: "I want to only create the VAE for this stage", integrating architectural taxonomy from reference configuration table and finalized design decisions.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Baseline VAE on CIFAR-10 (Priority: P1)

A researcher or ML practitioner wants to train a from-scratch Variational Autoencoder on CIFAR-10 ($32 \times 32 \times 3$) using the baseline architecture:
- 3-stage convolutional encoder ($[32, 64, 128]$ channels) with GroupNorm (8 groups) and LeakyReLU (0.2)
- Diagonal Gaussian posterior $q_\phi(z|x)$ with latent dimension $d=32$
- Standard isotropic Gaussian prior $p(z) = \mathcal{N}(0, I)$
- 3-stage transposed convolution decoder with `tanh` output bounding $[-1, 1]$
- Homoscedastic Gaussian likelihood with fixed $\sigma^2=1.0$ (MSE-equivalent ELBO)
- Exact analytical ELBO objective (reconstruction loss + analytical KL divergence)
- Modular component registry enabling future components to be plugged in via configuration

**Why this priority**: Establishes the foundational, mathematically validated generative pipeline on the benchmark dataset required by the project assignment before adding specialized VAE extensions.

**Independent Test**: Can be fully tested by training the baseline model on CIFAR-10 for a specified epoch count via the unified CLI or notebook runner, verifying that training and validation losses decrease monotonically, and validating that a clean checkpoint is serialized.

**Acceptance Scenarios**:

1. **Given** CIFAR-10 training data ($32 \times 32 \times 3$) scaled to $[-1, 1]$ and the baseline configuration, **When** training is executed on the local GPU (or CPU/Colab), **Then** the system computes the exact ELBO decomposed into reconstruction MSE/NLL and analytical KL divergence, tracks metrics per epoch, and saves the final and best checkpoints.
2. **Given** a trained model checkpoint and the CIFAR-10 test split, **When** evaluation is executed, **Then** the system reports reconstruction error and ELBO bounds.

---

### User Story 2 - Generate Synthetic CIFAR-10 Samples from Latent Prior (Priority: P2)

A practitioner wants to synthesize $32 \times 32 \times 3$ image samples by sampling 32-dimensional latent vectors directly from $\mathcal{N}(0, I)$ and passing them through the trained decoder, producing reproducible sample grids.

**Why this priority**: Validates that the decoder has learned a meaningful continuous mapping from the prior distribution to the image manifold.

**Independent Test**: Can be fully tested by loading a trained checkpoint, setting a fixed seed, generating a $8 \times 8$ (64 images) sample grid, and verifying output integrity and exact determinism across repeated executions.

**Acceptance Scenarios**:

1. **Given** a trained baseline checkpoint, **When** generation is executed with seed $S$ and count 64, **Then** the system outputs a formatted $8 \times 8$ image grid artifact in `artifacts/samples/`.
2. **Given** the same seed $S$ on consecutive runs, **Then** the generated pixel tensors are bitwise identical.

---

### User Story 3 - Latent Manifold Traversal and Pairwise Interpolation (Priority: P3)

An analyst wants to encode test image pairs into 32-dimensional latent space and compute spherical/linear interpolations, or traverse along individual latent axes, generating progressive visual strips to evaluate latent continuity, smoothness, and absence of mode collapse.

**Why this priority**: Essential qualitative diagnostic tool for generative quality and latent organization.

**Independent Test**: Can be fully tested by supplying two test images to the interpolation command and verifying the generation of an ordered sequence of intermediate images.

### Acceptance Scenarios:

1. **Given** two test CIFAR-10 images and an interpolation step count $K=10$, **When** interpolation is executed, **Then** the system produces a continuous $K$-frame transition image strip showing smooth semantic transformations.

---

### Edge Cases

- **Posterior Collapse ($D_{KL} \to 0$)**: If the encoder collapses to the prior and outputs constant latent representations, the system logs diagnostic warnings and tracks per-epoch KL divergence magnitude.
- **CUDA Out-Of-Memory on 4GB VRAM**: If batch size is set excessively high for the local Quadro T2000 GPU, the system provides clear configuration defaults (batch size 64/128) that safely fit within 4GB VRAM and fails gracefully with memory recommendations if exceeded.
- **Numerical Instability in Log-Variance**: In diagonal posterior modeling, unbounded $\log \sigma^2$ can produce NaNs. The system enforces numerical stability clamping: $\log \sigma^2 \in [-20, 2]$.
- **Corrupted or Incompatible Checkpoints**: If a user attempts to load a checkpoint with mismatched latent dimension or configuration, the system rejects it with an explicit schema mismatch error.

## Requirements *(mandatory)*

### Architectural Component Specifications & Registry Pattern

The VAE codebase MUST be architected with an extensible Registry Pattern across all 7 component categories so future components can be plugged in via configuration without altering core training or execution pipelines:

1. **Encoder Registry (`BaseEncoder`)**:
   - `cnn` (**Baseline**): 3-stage convolutional downsampling ($32 \times 32 \to 16 \times 16 \to 8 \times 8 \to 4 \times 4$) with channels $[32, 64, 128]$, GroupNorm (8 groups), and LeakyReLU (0.2). Flattened spatial representation ($128 \times 4 \times 4 = 2048$) projected via linear layers to distribution parameters.
   - `mlp` (*Extensible*): Dense layers for flattened inputs.
   - `resnet` (*Extensible*): Residual blocks with skip connections.
2. **Posterior Model Registry (`BasePosterior`)**:
   - `diagonal` (**Baseline**): Factorized Gaussian predicting $\mu \in \mathbb{R}^d$ and clamped $\log \sigma^2 \in \mathbb{R}^d$ ($2 \times d$ parameters), using the reparameterization trick $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$.
   - `full_covariance` (*Extensible*): Predicting $\mu$ and Cholesky matrix $L$.
3. **Latent Prior Registry (`BasePrior`)**:
   - `standard_gaussian` (**Baseline**): Fixed $\mathcal{N}(0, I)$ with closed-form analytical KL divergence: $-\frac{1}{2} \sum [1 + \log \sigma^2 - \mu^2 - \sigma^2]$.
   - `learnable_gaussian` (*Extensible*): Learnable $\mu_p, \sigma_p^2$.
   - `gmm` (*Extensible*): Gaussian Mixture Model prior.
4. **Decoder Backbone (`BaseDecoder`)**:
   - `cnn` (**Baseline**): Linear projection from $d$ to spatial feature map ($128 \times 4 \times 4 = 2048$), followed by 3-stage transposed convolution upsampling ($4 \times 4 \to 8 \times 8 \to 16 \times 16 \to 32 \times 32$) with GroupNorm and LeakyReLU, ending with a convolution layer and `tanh` activation generating reconstructions in $[-1, 1]$.
   - `mlp` (*Extensible*): Dense expansion from latent dimension.
5. **Likelihood Registry (`BaseLikelihood`)**:
   - `gaussian_homo` (**Baseline**): Continuous Gaussian likelihood with fixed noise variance $\sigma^2=1.0$ (proportional to MSE reconstruction loss $\frac{1}{2} \|x - \hat{x}\|^2$).
   - `bernoulli` (*Extensible*): Soft binary probability model.
   - `gaussian_hetero_image` (*Extensible*): Imagewise variance prediction.
   - `gaussian_hetero_pixel` (*Extensible*): Pixelwise spatial variance map.
6. **Loss Objectives Registry (`BaseLoss`)**:
   - `elbo` (**Baseline**): Standard Evidence Lower Bound ($\text{Reconstruction Loss} + D_{KL}$).
   - `beta_vae` (*Extensible*): Scaled $\beta \cdot D_{KL}$.
   - `beta_nll` (*Extensible*): Variance-stabilized NLL weighting.
   - `kl_free_bits` (*Extensible*): Clamped capacity per latent dimension.
7. **Training & Optimization Controls**:
   - Early stopping disabled by default, pluggable moving-window early stopping.
   - Configurable optimizer (Adam/AdamW), learning rate, batch size, and optional KL warmup schedule.

### Functional Requirements

- **FR-001**: System MUST instantiate models dynamically via a decorator-based component registry (`@register_encoder`, `@register_decoder`, `@register_posterior`, `@register_prior`, `@register_likelihood`, `@register_loss`) based on declarative YAML configuration files.
- **FR-002**: System MUST implement from foundational PyTorch tensor primitives the continuous Gaussian reparameterization trick $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ without external wrapper libraries.
- **FR-003**: System MUST compute the exact closed-form KL divergence between a diagonal Gaussian posterior $q_\phi(z|x)$ and standard normal prior $p(z)$: $-\frac{1}{2} \sum [1 + \log \sigma^2 - \mu^2 - \sigma^2]$.
- **FR-004**: System MUST support continuous image tensors for CIFAR-10 ($3 \times 32 \times 32$), normalized to $[-1, 1]$ and bounded by `tanh` in the decoder reconstruction head.
- **FR-005**: System MUST enforce deterministic execution across Python, NumPy, and PyTorch (CPU/CUDA) given a global seed argument.
- **FR-006**: System MUST provide a unified CLI interface powered by Typer supporting `train`, `evaluate`, `generate`, and `interpolate` commands with structured logging.
- **FR-007**: System MUST serialize checkpoints containing model weights, configuration schema, optimizer state, epoch index, and validation metrics, and restore them cleanly across runs.
- **FR-008**: System MUST execute automated tensor integrity tests validating dimensional preservation and non-zero gradient flow prior to training.

### Key Entities

- **Model Configuration**: Structured configuration specifying component keys (`cnn`, `diagonal`, `standard_gaussian`, etc.), latent dimension $d=32$, hyperparameters, and data pipeline options.
- **Component Registry**: Centralized factory mapping string identifiers to component implementations inheriting from base abstract contracts (`BaseEncoder`, `BaseDecoder`, `BasePosterior`, `BasePrior`, `BaseLikelihood`, `BaseLoss`).
- **Model Checkpoint**: Persistent state file storing model weights, optimizer parameters, metadata, and configuration provenance.
- **Training Artifacts**: Directory containing logs, metrics JSON, loss curves, reconstructed image comparisons, and generated sample grids for a specific run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Baseline CNN VAE on CIFAR-10 with $d=32$ trains on the local NVIDIA Quadro T2000 (4GB VRAM) at batch size 128 utilizing under 2.5 GB VRAM without out-of-memory errors.
- **SC-002**: Automated tensor integrity unit tests verify that forward and backward passes execute with non-zero gradients throughout all layers and without NaN values.
- **SC-003**: Training convergence verified by monotonic downward trend in validation ELBO across initial training epochs.
- **SC-004**: Synthetic sample generation of 64 images completes in under 2 seconds on GPU and under 5 seconds on CPU.
- **SC-005**: Checkpoints saved during training restore cleanly into the evaluation/generation commands and reproduce identical metric evaluations.

## Assumptions

- **Target Dataset**: CIFAR-10 ($32 \times 32 \times 3$, 10 classes, 50,000 train / 10,000 test images), downloaded automatically via standard torchvision dataset utilities.
- **Execution Interface**: Primary unified CLI (`src/cli/`) for local development, training, testing, evaluation, and inference.
- **Hardware Profile**: Local execution verified on Intel i7-10850H CPU and NVIDIA Quadro T2000 (4GB VRAM) with CUDA 12.8.
