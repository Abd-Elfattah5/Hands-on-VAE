# Data Model & Entity Specifications: Standalone VAE

**Feature**: `001-create-vae`
**Status**: Completed

This document defines the core data entities, schemas, in-memory structures, and serialization contracts for the modular VAE framework.

---

## 1. Core Domain Entities

```text
┌───────────────────────────┐         ┌───────────────────────────┐
│     ModelConfig           │         │         VAEOutput         │
├───────────────────────────┤         ├───────────────────────────┤
│ encoder: ComponentConfig  │         │ reconstruction: Tensor    │
│ posterior: ComponentConfig│───────► │ z: Tensor                 │
│ prior: ComponentConfig    │ builds  │ posterior: PosteriorDist  │
│ decoder: ComponentConfig  │         │ prior: PriorDist          │
│ likelihood: CompConfig    │         │ extra: dict[str, Any]     │
│ loss: ComponentConfig     │         └───────────────────────────┘
│ latent_dim: int           │                       │
└───────────────────────────┘                       ▼
              │                       ┌───────────────────────────┐
              │                       │         LossOutput        │
              ▼                       ├───────────────────────────┤
┌───────────────────────────┐         │ total_loss: Tensor        │
│      ModelCheckpoint      │         │ reconstruction_loss: float│
├───────────────────────────┤         │ kl_divergence: float      │
│ model_state_dict: dict    │         │ metrics: dict[str, float] │
│ optimizer_state_dict: dict│         └───────────────────────────┘
│ config: ModelConfig       │
│ epoch: int                │
│ step: int                 │
│ metrics: dict[str, float] │
└───────────────────────────┘
```

---

## 2. In-Memory Tensor & Distribution Contracts

### 2.1 `VAEOutput` (Forward Pass Return Container)

```python
from dataclasses import dataclass, field
from typing import Any
import torch

@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    """Reconstructed image tensor, shape [B, C, H, W], bounded in [-1, 1]."""

    z: torch.Tensor
    """Sampled latent representation, shape [B, d]."""

    posterior: Any
    """Posterior distribution instance implementing BasePosteriorDistribution."""

    prior: Any
    """Prior distribution instance implementing BasePriorDistribution."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Auxiliary outputs (e.g. spatial variance map for heteroscedastic models)."""
```

### 2.2 `LossOutput` (Objective Evaluation Container)

```python
@dataclass
class LossOutput:
    loss: torch.Tensor
    """Scalar loss tensor with active computational graph for backpropagation."""

    reconstruction_loss: torch.Tensor
    """Detached scalar value of negative log-likelihood / MSE reconstruction loss."""

    kl_divergence: torch.Tensor
    """Detached scalar value of analytical/sampled KL divergence."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Detailed decomposed metrics dictionary for logging (e.g. per-dimension KL)."""
```

---

## 3. Configuration Entities & Schemas

### 3.1 `ComponentConfig`

Each modular subcomponent in the 7 categories is defined by:
- `name`: String identifier corresponding to a registered decorator (e.g. `"cnn"`, `"diagonal"`, `"standard_gaussian"`, `"gaussian_homo"`, `"elbo"`).
- `params`: Arbitrary key-value parameters passed to the component constructor.

### 3.2 Top-Level `ExperimentConfig`

```yaml
experiment:
  name: "cifar10_baseline"
  seed: 42
  device: "auto" # "cuda", "cpu", or "auto"
  output_dir: "artifacts/runs/cifar10_baseline"

data:
  dataset: "cifar10"
  data_dir: "data"
  batch_size: 128
  num_workers: 4
  val_split: 0.1 # 10% of train set reserved for validation

model:
  latent_dim: 32
  in_channels: 3
  image_size: 32
  encoder:
    name: "cnn"
    channels: [32, 64, 128]
    num_groups: 8
    act_func: "leaky_relu"
    negative_slope: 0.2
  posterior:
    name: "diagonal"
    clamp_min: -20.0
    clamp_max: 2.0
  prior:
    name: "standard_gaussian"
  decoder:
    name: "cnn"
    channels: [128, 64, 32]
    num_groups: 8
    act_func: "leaky_relu"
    negative_slope: 0.2
    final_act: "tanh"
  likelihood:
    name: "gaussian_homo"
    fixed_sigma: 1.0
  loss:
    name: "elbo"
    beta: 1.0

training:
  epochs: 50
  lr: 0.001
  weight_decay: 1.0e-5
  scheduler:
    type: "cosine"
    min_lr: 1.0e-5
  gradient_clip_val: 5.0
  save_every: 5
  eval_every: 1
```

---

## 4. Persistent Checkpoint Schema (`.pt`)

Serialized model checkpoints saved to disk adhere to this explicit dictionary structure:

```python
{
    "format_version": "1.0",
    "epoch": int,
    "global_step": int,
    "model_state_dict": dict[str, torch.Tensor],
    "optimizer_state_dict": dict[str, Any],
    "scheduler_state_dict": dict[str, Any] | None,
    "config": dict[str, Any], # Complete serialized ExperimentConfig
    "metrics": {
        "val_total_loss": float,
        "val_recon_loss": float,
        "val_kl_div": float,
        "best_val_loss": float,
    },
    "provenance": {
        "torch_version": str,
        "cuda_available": bool,
        "timestamp": str, # ISO-8601
        "seed": int,
    }
}
```

---

## 5. Validation Rules & Constraints

1. **Dimensional Consistency**:
   - `encoder.channels[-1] * (image_size // (2 ** num_downsample_blocks)) ** 2` must equal the input dimension of the posterior projection layer.
   - For CIFAR-10 ($32 \times 32$) with 3 downsampling stages: $32 \to 16 \to 8 \to 4$. Spatial feature map is $4 \times 4$. If `channels[-1] = 128`, linear projection dimension is $128 \times 4 \times 4 = 2048$.
2. **Latent Space Sizing**:
   - `latent_dim >= 1` (default 32).
3. **Numerical Boundaries**:
   - Diagonal posterior $\log \sigma^2$ is bounded in $[-20.0, 2.0]$ to prevent numerical divergence in $\exp(0.5 \cdot \log \sigma^2)$.
4. **Target Observation Boundaries**:
   - Image pixel targets must reside in $[-1.0, 1.0]$. Values outside this range will raise an assertion error during batch collation.
