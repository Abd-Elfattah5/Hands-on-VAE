# Implementation Plan: 002-enhanced-vae

**Branch**: `002-enhanced-vae` | **Date**: September 25, 2026 | **Spec**: [specs/002-enhanced-vae/spec.md](spec.md)

**Input**: Feature specification from `specs/002-enhanced-vae/spec.md`.

---

## Summary

Implement an enhanced Variational Autoencoder (VAE) addressing the mathematical blurriness limitation of the homoscedastic $L_2$ baseline. Specifically, introducing a pixel-wise heteroscedastic Gaussian observation likelihood with $\beta$-NLL loss stabilization ($\beta = 0.5$) in `src/models/likelihoods/gaussian_hetero.py`, a dual-head transposed CNN decoder in `src/models/decoders/cnn_hetero.py`, and expanding the latent space bottleneck from $d=32$ to $d=128$. The enhanced model will be trained on CIFAR-10 for 50 epochs and benchmarked directly against the baseline FID ($169.02$).

---

## Technical Context

**Language/Version**: Python 3.12+  
**Primary Dependencies**: PyTorch 2.6+cu124, torchvision, typer, pyyaml, matplotlib, pillow, tqdm, scipy, scikit-learn  
**Storage**: Local `.pt` checkpoints with environment and seed provenance metadata  
**Testing**: pytest (unit tests in `tests/unit/`)  
**Target Platform**: Linux with NVIDIA Quadro T2000 (4GB VRAM)  
**Project Type**: CLI application and modular generative vision modeling library  
**Performance Goals**: Complete 50-epoch training on CIFAR-10 in under 45 minutes on GPU; achieve FID $< 169.02$  
**Constraints**: Peak GPU VRAM must stay strictly under 1.5 GB (measured benchmark: $240.78\text{ MB}$)  
**Scale/Scope**: 50,000 CIFAR-10 training images; 10,000 test evaluation samples; 5,000 generative Inception evaluation samples  

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Principle I (From-Scratch Mathematical Modeling)**: The heteroscedastic decoder and $\beta$-NLL likelihood are implemented directly from foundational tensor primitives. No black-box generative libraries used.
- [x] **Principle II (Deterministic Reproducibility)**: Global seeding across Python, NumPy, PyTorch CPU, and CUDA enforced; checkpoints record provenance metadata.
- [x] **Principle III (Standardized Benchmarking & Evaluation)**: Evaluated quantitatively on CIFAR-10 test partition using ELBO, reconstruction MSE, active latent units, FID, and Inception Score.
- [x] **Principle IV (Test-Driven Tensor & Mathematical Integrity)**: Pre-training integrity verification (`vae verify`) and automated unit test suite required before training.
- [x] **Principle V (CLI-Driven Pipeline & Observability)**: Integrated into unified CLI (`verify`, `train`, `evaluate`, `benchmark`, `generate`, `interpolate`, `plot-latent`).

---

## Project Structure

### Documentation (this feature)

```text
specs/002-enhanced-vae/
├── spec.md              # Feature specification and acceptance criteria
├── plan.md              # This implementation plan
├── research.md          # Architectural decisions (heteroscedasticity, beta-NLL, d=128)
├── data-model.md        # Entities, tensor shapes, and YAML schema
├── contracts/
│   └── component-interfaces.md # HeteroscedasticCNNDecoder and beta-NLL contracts
└── quickstart.md        # Runnable verification and benchmarking scenarios
```

### Source Code Modifications

```text
configs/
├── cifar10_enhanced.yaml        # 50-epoch enhanced experiment config (d=128, beta-NLL)

src/
├── models/
│   ├── decoders/
│   │   └── cnn_hetero.py        # HeteroscedasticCNNDecoder (mu and log_var heads)
│   ├── likelihoods/
│   │   └── gaussian_hetero.py   # HeteroscedasticGaussianLikelihood (beta-NLL)
│   └── vae.py                   # Support decoder tuple returns (mu, log_var_map)
└── configs/
    └── schema.py                # Schema validator support for cnn_hetero and gaussian_hetero

tests/
└── unit/
    └── test_heteroscedastic.py  # Unit tests for hetero decoder shapes and beta-NLL loss
```
