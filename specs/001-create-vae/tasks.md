# Tasks: Standalone Variational Autoencoder (VAE)

**Input**: Design documents from `/specs/001-create-vae/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/cli.md`, `contracts/component-interfaces.md`
**Tests**: Automated unit tests for tensor integrity, gradient flow, and loss formulation are mandated by Project Constitution Principle IV.
**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, packaging, configuration, and foundational utilities.

- [X] T001 Create Python package build configuration with CLI console script `vae = "src.cli.main:app"` and PyTorch/Typer dependencies in `pyproject.toml`
- [X] T002 [P] Create declarative experiment baseline configuration for CIFAR-10 ($d=32$, channels $[32, 64, 128]$, GroupNorm 8 groups, LeakyReLU 0.2, homoscedastic Gaussian likelihood $\sigma^2=1.0$, batch size 128) in `configs/cifar10_baseline.yaml`
- [X] T003 [P] Implement deterministic seeding utility controlling Python `random`, `numpy`, and PyTorch CPU/CUDA in `src/utils/seeding.py`
- [X] T004 [P] Implement structured console and file logging utility with ISO-8601 timestamps in `src/utils/logging.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data containers, abstract base classes, component registries, config validation, and CIFAR-10 data pipeline that MUST be complete before ANY user story can be implemented.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 [P] Implement typed data classes `VAEOutput` (reconstruction tensor bounded in $[-1, 1]$, latent $z$, posterior object, prior object, extra dict) and `LossOutput` (scalar loss, reconstruction loss, KL divergence, metrics dict) in `src/models/types.py`
- [X] T006 [P] Define Abstract Base Classes (`BaseEncoder`, `BaseDecoder`, `BasePosterior`, `BasePrior`, `BaseLikelihood`, `BaseLoss`) with typed abstract methods in `src/models/base.py`
- [X] T007 Implement decorator registries (`@register_encoder`, `@register_decoder`, `@register_posterior`, `@register_prior`, `@register_likelihood`, `@register_loss`) and dynamic `build_vae_from_config` factory in `src/models/registry.py`
- [X] T008 [P] Implement strict typed configuration schema and validator checking dimensional contracts (e.g. spatial downsampling to $4 \times 4$, group divisibility) in `src/configs/schema.py`
- [X] T009 [P] Implement CIFAR-10 dataset loading, 10% validation split, and normalization `Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))` to $[-1, 1]$ in `src/data/cifar10.py`

**Checkpoint**: Foundation ready - user story implementation can now begin.

---

## Phase 3: User Story 1 - Train Baseline VAE on CIFAR-10 (Priority: P1) 🎯 MVP

**Goal**: Build from-scratch VAE architecture, verify layer dimension contracts and non-zero gradient flow, train on CIFAR-10, track decomposed ELBO metrics, and serialize checkpoints.

**Independent Test**: `vae verify --config configs/cifar10_baseline.yaml` passes tensor integrity checks, `vae train` runs for 2 epochs and saves checkpoints, and `vae evaluate` outputs test loss and ELBO.

### Tests for User Story 1 ⚠️

- [X] T010 [P] [US1] Unit test for continuous Gaussian reparameterization $z = \mu + \sigma \odot \epsilon$ and non-zero gradient flow in `tests/unit/test_reparameterization.py`
- [X] T011 [P] [US1] Unit test for 3-stage CNN encoder/decoder dimensional preservation ($32 \to 16 \to 8 \to 4 \to 32$) and spatial shape contracts in `tests/unit/test_shapes.py`
- [X] T012 [P] [US1] Unit test for analytical ELBO loss calculation, closed-form KL divergence, and absence of NaNs in `tests/unit/test_loss.py`
- [X] T013 [P] [US1] Unit test for registry component lookup and dynamic model instantiation in `tests/unit/test_registry.py`

### Implementation for User Story 1

- [X] T014 [P] [US1] Implement 3-stage CNN encoder with channels $[32, 64, 128]$, `GroupNorm(num_groups=8)`, and `LeakyReLU(negative_slope=0.2)` producing 2048-dim feature vectors in `src/models/encoders/cnn.py`
- [X] T015 [P] [US1] Implement factorized diagonal Gaussian posterior predicting $\mu \in \mathbb{R}^{32}$ and $\log \sigma^2 \in \mathbb{R}^{32}$ with clamping `[-20.0, 2.0]` and reparameterization trick in `src/models/posteriors/diagonal.py`
- [X] T016 [P] [US1] Implement standard isotropic Gaussian prior $\mathcal{N}(0, I)$ with analytical closed-form KL divergence in `src/models/priors/standard_gaussian.py`
- [X] T017 [P] [US1] Implement 3-stage transposed convolution decoder with `GroupNorm(8)`, `LeakyReLU(0.2)`, and Conv2D + `tanh` output head bounding reconstructions to $[-1, 1]$ in `src/models/decoders/cnn.py`
- [X] T018 [P] [US1] Implement homoscedastic Gaussian likelihood with fixed $\sigma^2 = 1.0$ (proportional to MSE reconstruction loss $\frac{1}{2} \|x - \hat{x}\|^2$) in `src/models/likelihoods/gaussian_homo.py`
- [X] T019 [P] [US1] Implement analytical ELBO loss objective module computing negative log-likelihood MSE and analytical KL divergence in `src/models/losses/elbo.py`
- [X] T020 [US1] Implement composite `VAE` top-level module assembling registered subcomponents into a unified forward pass returning `VAEOutput` in `src/models/vae.py` (depends on T014-T019)
- [X] T021 [P] [US1] Implement checkpoint serializer and restorer managing `.pt` files (weights, optimizer, scheduler, config, metrics, provenance) in `src/training/checkpoint.py`
- [X] T022 [US1] Implement training and validation execution loop with per-epoch progress, loss tracking, and checkpoint saving in `src/training/trainer.py` (depends on T020, T021)
- [X] T023 [P] [US1] Implement quantitative test split evaluation computing test ELBO, reconstruction loss, and active latent units in `src/evaluation/evaluator.py`
- [X] T024 [US1] Implement `verify`, `train`, and `evaluate` CLI commands in `src/cli/main.py` (depends on T022, T023)

**Checkpoint**: User Story 1 is fully functional and testable as the standalone MVP.

---

## Phase 4: User Story 2 - Generate Synthetic CIFAR-10 Samples from Latent Prior (Priority: P2)

**Goal**: Draw 32-dimensional random vectors from $\mathcal{N}(0, I)$, decode them through the trained generator, un-normalize from $[-1, 1]$ to $[0, 1]$, and render reproducible $8 \times 8$ sample grids.

**Independent Test**: `vae generate --checkpoint ... --num-samples 64 --out artifacts/samples/sample_grid.png --seed 42` produces a visual grid, bit-for-bit identical across runs with the same seed.

### Tests for User Story 2 ⚠️

- [X] T025 [P] [US2] Unit test for deterministic prior sampling and image un-normalization in `tests/unit/test_generation.py`

### Implementation for User Story 2

- [X] T026 [US2] Implement sample synthesis and grid formatting utility ($8 \times 8$ square grid un-normalized to $[0, 1]$) in `src/evaluation/visualizer.py`
- [X] T027 [US2] Implement `generate` CLI command with `--num-samples`, `--out`, and `--seed` options in `src/cli/main.py` (depends on T026)

**Checkpoint**: User Stories 1 AND 2 are functional and verifiable independently.

---

## Phase 5: User Story 3 - Latent Manifold Traversal and Pairwise Interpolation (Priority: P3)

**Goal**: Encode two test CIFAR-10 images to $z_1, z_2 \in \mathbb{R}^{32}$, linearly/spherically interpolate across $K$ discrete steps, decode each point, and assemble a transition strip.

**Independent Test**: `vae interpolate --checkpoint ... --img1 ... --img2 ... --steps 10 --out artifacts/interpolations/walk.png` creates a 10-frame transition strip showing smooth visual transformation.

### Tests for User Story 3 ⚠️

- [X] T028 [P] [US3] Unit test for spherical/linear latent interpolation mathematical helper in `tests/unit/test_interpolation.py`

### Implementation for User Story 3

- [X] T029 [US3] Implement latent encoding, interpolation, and strip rendering functions in `src/evaluation/visualizer.py`
- [X] T030 [US3] Implement `interpolate` CLI command with `--img1`, `--img2`, `--steps`, and `--out` options in `src/cli/main.py` (depends on T029)

**Checkpoint**: All three user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories, test suites, and documentation.

- [X] T031 [P] Implement unit tests for checkpoint save, restore, and schema portability in `tests/unit/test_checkpoint.py`
- [X] T032 [P] Create test fixtures and synthetic batch tensor generators in `tests/conftest.py`
- [X] T033 Run full automated unit test suite `pytest tests/ -v` and verify 100% pass rate
- [X] T034 Execute quickstart validation scenarios from `specs/001-create-vae/quickstart.md` (verify, smoke train, evaluate, generate, interpolate)

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Phase 1: Setup (T001-T004)
        │
        ▼
Phase 2: Foundational (T005-T009) ── [BLOCKS ALL USER STORIES]
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
Phase 3: User Story 1 (T010-T024)   Phase 4: User Story 2 (T025-T027)
        │                                  │
        └─────────────────┬────────────────┘
                          ▼
                   Phase 5: User Story 3 (T028-T030)
                          │
                          ▼
                   Phase 6: Polish (T031-T034)
```

### User Story Dependencies

- **User Story 1 (P1)**: Can start immediately after Phase 2 (Foundational). Has no dependencies on other stories. Delivers the complete training and evaluation MVP.
- **User Story 2 (P2)**: Can start after Phase 2 (Foundational). Utilizes the trained checkpoint and decoder from US1 to synthesize image grids.
- **User Story 3 (P3)**: Can start after Phase 2 (Foundational). Utilizes the trained checkpoint, encoder, and decoder to interpolate between images.

### Parallel Opportunities

- **Phase 1 (Setup)**: `T002`, `T003`, and `T004` can execute in parallel after `T001`.
- **Phase 2 (Foundational)**: `T005`, `T006`, `T008`, and `T009` can execute in parallel. `T007` follows `T006`.
- **Phase 3 (US1 Tests)**: `T010`, `T011`, `T012`, and `T013` can all execute in parallel before implementation.
- **Phase 3 (US1 Models)**: `T014`, `T015`, `T016`, `T017`, `T018`, `T019`, and `T021` can all execute in parallel.
- **Phase 4 & 5**: Tests and visualization methods can execute in parallel once the core composite VAE model exists.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (`T001-T004`).
2. Complete Phase 2: Foundational (`T005-T009`).
3. Complete Phase 3: User Story 1 (`T010-T024`).
4. **STOP and VALIDATE**: Run `vae verify --config configs/cifar10_baseline.yaml`, run 2-epoch smoke training, and evaluate test ELBO. The system is a fully functional MVP.

### Incremental Delivery

1. **Increment 1 (MVP)**: Trainable VAE on CIFAR-10 with test ELBO metrics (`vae train`, `vae evaluate`).
2. **Increment 2**: Prior sampling and visual sample grid generation (`vae generate`).
3. **Increment 3**: Latent space continuity inspection and interpolation strips (`vae interpolate`).
4. **Increment 4**: End-to-end polish, quickstart validation, and automated test regression suite (`T031-T034`).

---

## Phase 7: Convergence

- [X] T035 [CRITICAL] Implement unit test for continuous Gaussian reparameterization trick and non-zero gradient flow in `tests/unit/test_reparameterization.py` per Constitution IV, SC-002 (missing)
- [X] T036 [CRITICAL] Implement unit test for 3-stage CNN encoder/decoder dimensional preservation ($32 \to 16 \to 8 \to 4 \to 32$) in `tests/unit/test_shapes.py` per Constitution IV, SC-002 (missing)
- [X] T037 [CRITICAL] Implement unit test for analytical ELBO loss calculation, closed-form KL divergence, and absence of NaNs in `tests/unit/test_loss.py` per Constitution IV, FR-003 (missing)
- [X] T038 [CRITICAL] Implement unit test for registry component lookup and dynamic model instantiation in `tests/unit/test_registry.py` per FR-001 (missing)
- [X] T039 [CRITICAL] Implement 3-stage CNN encoder with channels $[32, 64, 128]$, `GroupNorm(num_groups=8)`, and `LeakyReLU(0.2)` producing 2048-dim feature representations in `src/models/encoders/cnn.py` per Constitution I, FR-001 (missing)
- [X] T040 [CRITICAL] Implement factorized diagonal Gaussian posterior predicting $\mu \in \mathbb{R}^{32}$ and $\log \sigma^2 \in \mathbb{R}^{32}$ clamped to $[-20.0, 2.0]$ with reparameterization in `src/models/posteriors/diagonal.py` per Constitution I, FR-002 (missing)
- [X] T041 [CRITICAL] Implement standard isotropic Gaussian prior $\mathcal{N}(0, I)$ with closed-form analytical KL divergence in `src/models/priors/standard_gaussian.py` per Constitution I, FR-003 (missing)
- [X] T042 [CRITICAL] Implement 3-stage transposed convolution decoder with `GroupNorm(8)`, `LeakyReLU(0.2)`, and Conv2D + `tanh` output head in `src/models/decoders/cnn.py` per Constitution I, FR-004 (missing)
- [X] T043 [CRITICAL] Implement homoscedastic Gaussian likelihood with fixed $\sigma^2 = 1.0$ in `src/models/likelihoods/gaussian_homo.py` per Constitution I, FR-001 (missing)
- [X] T044 [CRITICAL] Implement analytical ELBO loss objective module computing negative log-likelihood MSE and analytical KL divergence in `src/models/losses/elbo.py` per Constitution I, FR-003 (missing)
- [X] T045 [CRITICAL] Implement composite `VAE` top-level module assembling registered subcomponents into unified forward pass in `src/models/vae.py` per Constitution I, FR-001 (missing)
- [X] T046 [CRITICAL] Implement checkpoint serializer and restorer managing `.pt` files with complete provenance metadata in `src/training/checkpoint.py` per Constitution II, FR-007 (missing)
- [X] T047 [CRITICAL] Implement training and validation execution loop with per-epoch loss tracking and checkpoint saving in `src/training/trainer.py` per US1/AC1 (missing)
- [X] T048 [CRITICAL] Implement quantitative test split evaluation computing test ELBO, reconstruction loss, and active latent units in `src/evaluation/evaluator.py` per Constitution III, US1/AC2 (missing)
- [X] T049 [CRITICAL] Implement CLI commands `verify`, `train`, and `evaluate` in `src/cli/main.py` per Constitution V, FR-006, FR-008 (missing)
- [X] T050 Implement unit test for deterministic prior sampling and image un-normalization in `tests/unit/test_generation.py` per US2/AC2 (missing)
- [X] T051 Implement sample synthesis and $8 \times 8$ grid formatting utility in `src/evaluation/visualizer.py` per Constitution III, US2/AC1, SC-004 (missing)
- [X] T052 Implement CLI command `generate` with `--num-samples`, `--out`, and `--seed` options in `src/cli/main.py` per Constitution V, FR-006 (missing)
- [X] T053 Implement unit test for latent interpolation mathematical helper in `tests/unit/test_interpolation.py` per US3/AC1 (missing)
- [X] T054 Implement latent encoding, pairwise interpolation, and transition strip rendering in `src/evaluation/visualizer.py` per Constitution III, US3/AC1 (missing)
- [X] T055 Implement CLI command `interpolate` with `--img1`, `--img2`, `--steps`, and `--out` options in `src/cli/main.py` per Constitution V, FR-006 (missing)
- [X] T056 Implement unit tests for checkpoint save, restore, and schema portability in `tests/unit/test_checkpoint.py` per FR-007, SC-005 (missing)
- [X] T057 Execute full automated unit test suite `pytest tests/ -v` and verify 100% pass rate per Constitution IV, SC-002 (missing)
- [X] T058 Execute quickstart validation scenarios from `specs/001-create-vae/quickstart.md` per plan: quickstart validation (missing)

---

## Phase 8: Benchmarking, Latent Diagnostics & Deliverables (GenCV003)

**Purpose**: Fulfill 100% of the VAE requirements from `GenCV003.pdf`, including high-resolution visual rendering, pure synthetic latent space diagnostics, quantitative FID/IS benchmarking, formal technical report Deliverable (a), and reproduction guide Deliverable (b).

### High-Resolution Rendering & Pure Synthetic Latent Diagnostics

- [X] T059 [P] Implement high-resolution Lanczos/bicubic canvas upscaling (`--upscale` parameter, $4\times \to 128 \times 128$ per tile, yielding $1024 \times 1024$ grids) in `src/evaluation/visualizer.py`
- [X] T060 [P] Implement side-by-side original vs. reconstructed image comparison gallery utility with PSNR/MSE metrics in `src/evaluation/visualizer.py`
- [X] T061 [P] Implement pure synthetic 2D bilinear interpolation from prior $\mathcal{N}(0, I)$ without real image encoding in `src/evaluation/visualizer.py`
- [X] T062 [P] Implement 1D latent coordinate traversal (sweeping active axes $z_j \in [-3.0, +3.0]$ with all other dimensions clamped to $0$) in `src/evaluation/visualizer.py`

### Latent Space Dimensionality Reduction & Manifold Analysis

- [X] T063 [P] Implement t-SNE and PCA dimensionality reduction projecting 32D posterior means $\mu(x)$ to 2D with CIFAR-10 class labels and prior overlay in `src/evaluation/latent_analysis.py`
- [X] T064 [P] Implement 2D latent manifold meshgrid traversal across top 2 active latent dimensions $[-2.5, +2.5] \times [-2.5, +2.5]$ in `src/evaluation/latent_analysis.py`

### Quantitative Benchmarking Pipeline (FID & Inception Score)

- [X] T065 [P] Implement Inception Score (IS) computation over synthetic VAE samples in `src/evaluation/metrics.py`
- [X] T066 [P] Implement Fréchet Inception Distance (FID) computation between real CIFAR-10 test partition and synthetic VAE samples in `src/evaluation/metrics.py`

### CLI Integration

- [X] T067 Implement `vae benchmark` CLI command computing FID, IS, ELBO, and active units saving to `artifacts/eval/benchmark_metrics.json` in `src/cli/main.py`
- [X] T068 Implement `vae plot-latent` CLI command generating t-SNE, PCA, and 2D manifold traversals to `artifacts/eval/` in `src/cli/main.py`
- [X] T069 Update `vae interpolate` and `vae generate` CLI commands with `--upscale` and `--synthetic` flags in `src/cli/main.py`

### Unit Tests

- [X] T070 [P] Implement unit tests for Inception Score and FID computation pipelines in `tests/unit/test_metrics.py`
- [X] T071 [P] Implement unit tests for t-SNE/PCA projections and coordinate sweep utilities in `tests/unit/test_latent_analysis.py`

### Documentation & Deliverables (GenCV003)

- [X] T072 Author comprehensive Baseline VAE Technical Report documenting mathematical derivations, 50-epoch loss curves, quantitative metrics (FID, IS, ELBO), qualitative high-res plates, and failure mode analysis in `docs/reports/001-baseline-vae-report.md` per Deliverable (a)
- [X] T073 Update root `README.md` with project overview, mathematical foundation, installation instructions, and CLI reproduction guide in `README.md` per Deliverable (b)
