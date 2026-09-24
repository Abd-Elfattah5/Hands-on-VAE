<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.1.0
- List of modified principles:
  - Scoped project exclusively to Variational Autoencoder (VAE) architecture and benchmarking.
-->

# Hands-on VAE Constitution

## Core Principles

### I. From-Scratch Mathematical Modeling
The Variational Autoencoder architecture MUST be implemented from foundational tensor operations and neural network primitives. High-level wrapper libraries or pre-packaged model pipelines MUST NOT be used for the generative backbones. Mathematical mechanisms—including the encoder-decoder topology, Gaussian reparameterization trick, and Evidence Lower Bound (ELBO) loss decomposition—must be explicitly authored, mathematically grounded, and self-contained within the project codebase.

### II. Deterministic Reproducibility
All experimental pipelines, data splits, and model training routines MUST be deterministic and reproducible. Global random seeds across all active runtimes (Python `random`, `numpy`, PyTorch CPU/CUDA) must be strictly controlled and exposed via configuration. Environment specifications, data preprocessing pipelines, hyperparameters, and model checkpoints MUST be versioned and saved with unambiguous provenance so that any run, quantitative metric, or qualitative visual can be recreated on demand.

### III. Standardized Benchmarking & Evaluation
Model evaluation MUST be rigorous and conducted under standardized conditions on the CIFAR-10 public benchmark dataset using consistent training/validation/testing partitions. Evaluation MUST incorporate both quantitative rigor and qualitative inspection:
- Quantitative metrics MUST include standard generative performance indicators such as Fréchet Inception Distance (FID) and Inception Score (IS), alongside reconstruction loss and log-likelihood/ELBO bounds.
- Qualitative assessments MUST systematically capture visual fidelity, diversity, mode collapse diagnostics, and latent space continuity (e.g., spherical/linear interpolations and dimensional traversals).

### IV. Test-Driven Tensor & Mathematical Integrity
All neural network modules and custom loss functions MUST be verified through automated unit tests before model training commences. Verification MUST explicitly check:
- Tensor dimension preservation and spatial transformation contracts through all network layers.
- Non-zero gradient flow through the reparameterization trick.
- Numerical stability of loss formulations (e.g., preventing log-variance explosion, NaN gradients, or KL divergence collapse).

### V. CLI-Driven Pipeline & Artifact Observability
All operational workflows—dataset downloading/preprocessing, model training, checkpoint evaluation, image generation, and metric aggregation—MUST be exposed through modular Command Line Interfaces. Pipelines MUST adhere to structured I/O principles: runtime configurations supplied via CLI arguments or structured config files, machine-readable logs and metrics emitted in JSON/CSV format alongside standard console output, and visual artifacts (plots, sample grids, interpolation strips) automatically organized in designated artifact directories.

## Technical Stack & Architectural Constraints

- **Language & Core Framework**: Python 3.12+ using PyTorch for tensor computation and automatic differentiation.
- **Dependency Isolation**: All third-party dependencies MUST be managed within an isolated virtual environment (`VAE/`) and recorded in reproducible dependency manifests.
- **Architectural Separation**: Code MUST be modularly separated into distinct packages:
  - `models/`: Self-contained VAE architectural implementations and component registries.
  - `data/`: Dataset loading, augmentation, and normalization pipelines.
  - `training/`: Training loops, optimizers, learning rate schedulers, and checkpoint handlers.
  - `evaluation/`: Metric computation (FID, Inception Score) and qualitative visual sampling.
  - `utils/`: Deterministic seeding, configuration loading, and logging utilities.
- **Prohibited Patterns**: No monolithic unmaintainable scripts; training loops and model definitions must not be entangled in unversioned interactive notebooks without corresponding tested library modules.

## Experimental Workflow & Quality Gates

1. **Pre-Training Gate**: Unit tests for model forward passes, loss computations, and backward gradient passes MUST pass prior to launching training experiments.
2. **Experiment Tracking**: Training loss, validation loss, and ELBO components (reconstruction vs. KL divergence) MUST be tracked and saved per epoch.
3. **Evaluation Gate**: Final checkpoints MUST undergo automated quantitative evaluation on the test split, producing timestamped metric summaries.
4. **Deliverable Reporting**: Analyses MUST culminate in a comprehensive technical report documenting theoretical formulation, sample generation speed, computational complexity, sample quality, and distribution coverage.

## Governance

This Constitution serves as the authoritative standard for architectural, empirical, and coding decisions in the project. Any deviation from these principles requires an explicit amendment.

- **Amendment Process**: Proposed amendments must be documented with clear technical rationale.
- **Versioning Policy**: Semantic Versioning (`MAJOR.MINOR.PATCH`).
- **Compliance**: All pull requests, code reviews, and project milestones must verify compliance with this constitution before merging or release.

**Version**: 1.1.0 | **Ratified**: 2026-09-23 | **Last Amended**: 2026-09-23
