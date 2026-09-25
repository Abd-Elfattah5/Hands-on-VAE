# Data Model & Component Contracts: 002-enhanced-vae

## 1. Architectural Entities

```
┌────────────────────────────────────────────────────────┐
│                   HeteroscedasticCNNDecoder            │
├────────────────────────────────────────────────────────┤
│ latent_dim: int = 128                                  │
│ channels: Sequence[int] = (128, 64, 32)                │
│ out_channels: int = 3                                  │
│ image_size: int = 32                                   │
│ num_groups: int = 8                                    │
│ act_func: str = "leaky_relu"                           │
│ clamp_min: float = -10.0                               │
│ clamp_max: float = 5.0                                 │
├────────────────────────────────────────────────────────┤
│ forward(z: Tensor) -> tuple[Tensor, Tensor]            │
│   ├── mu: Tensor [B, 3, 32, 32] ∈ [-1, 1]              │
│   └── log_var: Tensor [B, 3, 32, 32] ∈ [-10, 5]        │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                        VAEOutput                       │
├────────────────────────────────────────────────────────┤
│ reconstruction: Tensor [B, 3, 32, 32] ∈ [-1, 1]        │
│ z: Tensor [B, 128]                                     │
│ posterior: DiagonalGaussianDistribution                │
│ prior: StandardGaussianPrior                           │
│ extra: dict[str, Any]                                  │
│   └── "log_var_map": Tensor [B, 3, 32, 32]            │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             HeteroscedasticGaussianLikelihood          │
├────────────────────────────────────────────────────────┤
│ beta_nll: float = 0.5                                  │
├────────────────────────────────────────────────────────┤
│ negative_log_likelihood(recon, target, extra) -> Tensor│
│   └── returns beta-NLL loss per sample [B]             │
└────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Contracts

### 2.1 Decoder Dual Output
Given latent code $z \in \mathbb{R}^{B \times 128}$:
1. Spatial feature expansion:
   $$h_0 = \text{Reshape}(\text{Linear}(z), [B, 128, 4, 4])$$
2. 3-stage transposed convolution upsampling:
   $$h_3 = \text{DeconvNet}(h_0) \in \mathbb{R}^{B \times 32 \times 32 \times 32}$$
3. Dual prediction heads:
   $$\mu(z) = \tanh(\text{Conv2D}_{\mu}(h_3)) \in [-1.0, 1.0]^{B \times 3 \times 32 \times 32}$$
   $$\log \sigma_{\text{obs}}^2(z) = \text{clamp}(\text{Conv2D}_{\sigma}(h_3), \text{min}=-10.0, \text{max}=5.0)$$

### 2.2 Beta-NLL Formulation
Given prediction $\mu$, target image $x \in [-1, 1]$, and variance $\sigma^2 = \exp(\log \sigma^2)$:
$$\text{Squared Error: } \text{SE}_i = (x_i - \mu_i)^2$$
$$\text{Weighting factor: } w_i = (\sigma_i^2)^\beta_{\text{detached}}$$
$$\mathcal{L}_{\beta\text{-NLL}} = \frac{1}{2} \sum_{i=1}^{3072} \left[ \frac{\text{SE}_i}{\sigma_i^2} + \log \sigma_i^2 \right] \cdot w_i$$

---

## 3. Declarative Configuration Schema (`configs/cifar10_enhanced.yaml`)

```yaml
experiment:
  name: "cifar10_enhanced"
  seed: 42
  device: "auto"
  output_dir: "artifacts/runs/cifar10_enhanced"

data:
  dataset: "cifar10"
  data_dir: "data"
  batch_size: 128
  num_workers: 4
  val_split: 0.1

model:
  latent_dim: 128
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
    name: "cnn_hetero"
    channels: [128, 64, 32]
    num_groups: 8
    act_func: "leaky_relu"
    negative_slope: 0.2
    final_act: "tanh"
    clamp_min: -10.0
    clamp_max: 5.0
  likelihood:
    name: "gaussian_hetero"
    beta_nll: 0.5
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
