# Tasks: 002-enhanced-vae

**Input**: Design documents from `specs/002-enhanced-vae/` (`spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/component-interfaces.md`, `quickstart.md`)

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/component-interfaces.md`

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Configuration & Schema Validation)

**Purpose**: Establish declarative configuration and schema validation for heteroscedastic likelihood and expanded latent space.

- [ ] T001 [P] Create declarative enhanced configuration in `configs/cifar10_enhanced.yaml` specifying `latent_dim: 128`, `decoder.name: "cnn_hetero"`, and `likelihood.name: "gaussian_hetero"` with `beta_nll: 0.5`
- [ ] T002 [P] Extend configuration schema validator in `src/configs/schema.py` to validate `cnn_hetero` decoder clamp ranges and `gaussian_hetero` likelihood parameters

---

## Phase 2: Foundational (Interface Adaptors & Contracts)

**Purpose**: Adapt base likelihood and composite VAE contracts to support dual-head decoder outputs and auxiliary uncertainty maps.

- [ ] T003 [P] Update `BaseLikelihood.negative_log_likelihood` contract in `src/models/base.py` to accept optional `extra: Optional[dict[str, Any]] = None`
- [ ] T004 [P] Update `ELBOLoss.forward` in `src/models/losses/elbo.py` to pass `output.extra` into `self.likelihood.negative_log_likelihood`
- [ ] T005 Update composite `VAE` module in `src/models/vae.py` to intercept dual decoder output `(reconstruction, log_var_map)` and populate `output.extra["log_var_map"]`

**Checkpoint**: Foundation ready — model component implementation and testing can now proceed.

---

## Phase 3: User Story 1 - Heteroscedastic Likelihood & Beta-NLL (Priority: P1) 🎯 MVP

**Goal**: Implement dual-head CNN decoder predicting both mean $\mu(z)$ and log-variance map $\log \sigma_{\text{obs}}^2(z)$, coupled with $\beta$-NLL likelihood loss to mitigate $L_2$ conditional mean blurriness.

**Independent Test**: `vae verify --config configs/cifar10_enhanced.yaml` verifies non-zero gradients across both prediction heads and finite loss.

### Tests for User Story 1 ⚠️

- [ ] T006 [P] [US1] Unit test for `HeteroscedasticCNNDecoder` dual output shapes and clamp range `[-10.0, 5.0]` in `tests/unit/test_heteroscedastic.py`
- [ ] T007 [P] [US1] Unit test for `HeteroscedasticGaussianLikelihood` $\beta$-NLL loss calculation and gradient stabilization in `tests/unit/test_heteroscedastic.py`

### Implementation for User Story 1

- [ ] T008 [P] [US1] Implement `HeteroscedasticCNNDecoder` with dual prediction heads (`head_mu` with `tanh`, `head_log_var` clamped to `[-10.0, 5.0]`) in `src/models/decoders/cnn_hetero.py`
- [ ] T009 [P] [US1] Implement `HeteroscedasticGaussianLikelihood` computing $\beta$-NLL loss with $(\sigma^{2\beta})_{\text{detached}}$ weighting in `src/models/likelihoods/gaussian_hetero.py`
- [ ] T010 [US1] Register `"cnn_hetero"` and `"gaussian_hetero"` in `src/models/registry.py` and export in package `__init__.py` files
- [ ] T011 [US1] Execute pre-training architecture integrity verification via `vae verify --config configs/cifar10_enhanced.yaml`

**Checkpoint**: At this point, User Story 1 is verified and fully functional as the standalone MVP.

---

## Phase 4: User Story 2 - Expanded Latent Representation Capacity ($d=128$) (Priority: P2)

**Goal**: Scale the latent bottleneck capacity from 32 to 128 dimensions to capture high-frequency visual details without bottleneck saturation.

**Independent Test**: Train model for 50 epochs and verify active units $A_z > 32$ with lower reconstruction loss than the baseline.

### Tests for User Story 2 ⚠️

- [ ] T012 [P] [US2] Unit test verifying 128-dimensional latent space forward pass, reparameterization, and shape preservation in `tests/unit/test_heteroscedastic.py`

### Implementation for User Story 2

- [ ] T013 [US2] Execute 50-epoch training on CIFAR-10 via `vae train --config configs/cifar10_enhanced.yaml --epochs 50 --batch-size 128`

**Checkpoint**: Enhanced model trained for 50 epochs with checkpoints and trajectories saved in `artifacts/runs/cifar10_enhanced/`.

---

## Phase 5: User Story 3 - Comparative Quantitative & Qualitative Benchmarking (Priority: P3)

**Goal**: Execute comprehensive comparative benchmarking evaluating FID, Inception Score, test ELBO, PSNR, sample grid synthesis, and 2D manifold interpolations against the baseline.

**Independent Test**: `vae benchmark` achieves FID strictly lower than baseline FID of $169.02$.

### Implementation for User Story 3

- [ ] T014 [US3] Execute test split quantitative evaluation via `vae evaluate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt`
- [ ] T015 [US3] Execute quantitative benchmarking via `vae benchmark --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --num-samples 5000 --out artifacts/eval_enhanced/benchmark_metrics.json`
- [ ] T016 [US3] Generate high-resolution $1024 \times 1024$ sample grid via `vae generate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --num-samples 64 --out artifacts/samples/enhanced_sample_grid_1024.png --upscale 4`
- [ ] T017 [US3] Generate pure synthetic 2D prior manifold interpolation grid via `vae interpolate --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --synthetic --steps 8 --out artifacts/interpolations/enhanced_synthetic_grid_2d.png --upscale 4`
- [ ] T018 [US3] Generate comprehensive latent space diagnostics (t-SNE, PCA, 2D manifold, 1D sweeps, reconstruction gallery) via `vae plot-latent --checkpoint artifacts/runs/cifar10_enhanced/best_checkpoint.pt --out-dir artifacts/eval_enhanced`

**Checkpoint**: All quantitative benchmark metrics and qualitative visual plates generated and persisted.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Formalize comparative findings, update project documentation, and ensure 100% test suite pass rate.

- [ ] T019 [P] Author comparative Technical Report in `docs/reports/002-enhanced-vae-report.md` analyzing baseline vs. enhanced results (FID, PSNR, edge sharpness, uncertainty maps)
- [ ] T020 [P] Update root `README.md` benchmark summary table comparing Baseline ($d=32$, homoscedastic) vs. Enhanced ($d=128$, $\beta$-NLL heteroscedastic)
- [ ] T021 Run full automated test suite `pytest tests/ -v` verifying 100% pass rate with zero regressions

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Phase 1 completion — blocks all User Stories.
- **User Story 1 (Phase 3)**: Depends on Phase 2 completion — delivers testable MVP.
- **User Story 2 (Phase 4)**: Depends on Phase 3 completion — executes 50-epoch training.
- **User Story 3 (Phase 5)**: Depends on Phase 4 completion — evaluates trained checkpoints.
- **Polish (Phase 6)**: Depends on Phase 5 completion — documents results and verifies regression suite.

### Parallel Opportunities

- `T001` and `T002` can execute in parallel.
- `T003` and `T004` can execute in parallel.
- `T006`, `T007`, `T008`, `T009` can execute in parallel.
- `T019` and `T020` can execute in parallel.
