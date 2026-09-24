# Research & Architectural Decisions: Standalone VAE

**Feature**: `001-create-vae`
**Status**: Completed

This document captures all research findings, technology choices, and architectural trade-offs resolved during the planning phase for the from-scratch Variational Autoencoder.

---

## 1. Primary Benchmark Dataset & Preprocessing

- **Decision**: Target **CIFAR-10** ($32 \times 32 \times 3$, 50,000 train / 10,000 test images), with pixel normalization scaled to $[-1, 1]$ via `transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`.
- **Rationale**:
  - Direct alignment with the project assignment mandate (`GenCV003.pdf`) for public benchmark evaluation using Fréchet Inception Distance (FID) and Inception Score (IS).
  - Scaled range $[-1, 1]$ is zero-centered, providing stable numerical gradients for convolutional neural networks and directly matching the bounded output range of the `tanh` activation function on the decoder output layer.
- **Alternatives Considered**:
  - *MNIST / Fashion-MNIST* ($28 \times 28 \times 1$): Fast to train, but grayscale single-channel does not represent realistic generative CV benchmarks and does not align cleanly with standard pretrained Inception feature extractors for FID.
  - *CelebA* ($64 \times 64 \times 3$): Excellent latent facial semantics, but higher download footprint and training time on local 4GB GPU.

---

## 2. Convolutional Backbone Topology & Normalization

- **Decision**: 
  - **Encoder**: 3 convolutional downsampling blocks with stride 2 ($32 \times 32 \to 16 \times 16 \to 8 \times 8 \to 4 \times 4$), channel progression $[32, 64, 128]$, `GroupNorm` (8 groups), and `LeakyReLU(negative_slope=0.2)`. Flattened representation ($128 \times 4 \times 4 = 2048$) is linearly projected to $2 \times d$ variational parameters.
  - **Decoder**: Linear projection from latent space $d$ to spatial feature map ($128 \times 4 \times 4 = 2048$), followed by 3 transposed convolution stages (`ConvTranspose2d` with stride 2, kernel size 4, padding 1) expanding spatial dimensions ($4 \times 4 \to 8 \times 8 \to 16 \times 16 \to 32 \times 32$), with `GroupNorm` and `LeakyReLU(0.2)`, terminating in a Conv2D projection to 3 channels with `tanh` activation.
- **Rationale**:
  - `GroupNorm` divides channels into groups independently per sample. In generative models, standard `BatchNorm2d` introduces batch-level dependencies that cause mismatch during single-sample generation and evaluation. `GroupNorm` maintains high training stability without batch-size dependence.
  - `LeakyReLU(0.2)` avoids dying neuron regimes during early stochastic exploration.
  - Memory consumption for this topology on CIFAR-10 with batch size 128 is approximately ~800MB–1.2GB, well within the 4GB VRAM capacity of the NVIDIA Quadro T2000.
- **Alternatives Considered**:
  - *BatchNorm2d*: Rejected due to batch-dependency artifacts during test/generation time.
  - *No Normalization*: Viable, but slows down convergence compared to GroupNorm.
  - *Bilinear Upsampling + Conv2D*: Modular decoder interface is preserved so this can be plugged in if transposed convolutions exhibit checkerboard artifacts.

---

## 3. Variational Posterior & Latent Prior Formulation

- **Decision**: 
  - **Posterior $q_\phi(z|x)$**: Factorized diagonal Gaussian parameterizing mean $\mu \in \mathbb{R}^d$ and $\log \sigma^2 \in \mathbb{R}^d$ with $d=32$. Differentiable sampling via the reparameterization trick $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ and $\sigma = \exp(0.5 \cdot \log \sigma^2)$. Log-variance is clamped to $[-20, 2]$ to prevent numerical overflow/underflow.
  - **Prior $p(z)$**: Fixed standard isotropic normal distribution $\mathcal{N}(0, I)$.
  - **KL Divergence**: Analytical closed-form solution:
    $$D_{KL}(q_\phi(z|x) \parallel p(z)) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$
- **Rationale**:
  - Closed-form analytical KL evaluation exhibits significantly lower gradient variance than Monte Carlo sampled KL divergence.
  - Default latent dimension $d=32$ provides sufficient capacity for $32 \times 32 \times 3$ natural images while preventing trivial identity mappings and maintaining structured latent space geometry.
- **Alternatives Considered**:
  - *Full Covariance Gaussian ($L L^T$)*: Captures correlated latents, but adds $O(d^2)$ parameters; deferred as an extensible component plugin.
  - *Gaussian Mixture Model (GMM) Prior*: Mitigates prior holes in multimodal data, but requires Monte Carlo KL or variational upper bounds; deferred as an extensible component plugin.

---

## 4. Observation Likelihood Model & Loss Objective

- **Decision**:
  - **Likelihood $p_\theta(x|z)$**: Homoscedastic Gaussian distribution $\mathcal{N}(\mu_\theta(z), \sigma_{obs}^2 I)$ with fixed noise scale $\sigma_{obs}^2 = 1.0$.
  - **Loss Objective**: Evidence Lower Bound (ELBO) minimized as negative log-likelihood:
    $$\mathcal{L}_{ELBO} = \frac{1}{2 \sigma_{obs}^2} \|x - \hat{x}\|_2^2 + D_{KL}(q_\phi(z|x) \parallel p(z))$$
    Reconstruction error is averaged per pixel and batch, tracking reconstruction MSE and KL divergence as separate diagnostic metrics.
- **Rationale**:
  - Continuous image pixels in $[-1, 1]$ are mathematically grounded under a Gaussian observation density. With fixed $\sigma_{obs}^2 = 1.0$, negative log-likelihood is directly proportional to Mean Squared Error without optimization instabilities.
- **Alternatives Considered**:
  - *Bernoulli Likelihood*: Requires pixel normalization to $[0, 1]$; mathematically formulated for binary probabilities rather than continuous RGB color signals.
  - *Heteroscedastic Gaussian (Pixelwise)*: Predicts per-pixel noise $\sigma_{ij}$, but requires $\beta$-NLL stabilization to prevent the model from trivializing loss by blowing up predicted variance on high-frequency edges.

---

## 5. Software Architecture: Decorator Registry & `VAEOutput` Contract

- **Decision**:
  - Implement dynamic decorator registries across all 7 architectural categories:
    - `@register_encoder` (`BaseEncoder`)
    - `@register_decoder` (`BaseDecoder`)
    - `@register_posterior` (`BasePosterior`)
    - `@register_prior` (`BasePrior`)
    - `@register_likelihood` (`BaseLikelihood`)
    - `@register_loss` (`BaseLoss`)
  - Standardize model forward pass output via a typed `VAEOutput` dataclass:
    ```python
    @dataclass
    class VAEOutput:
        reconstruction: torch.Tensor
        z: torch.Tensor
        posterior: BasePosteriorDistribution
        prior: BasePriorDistribution
        extra: dict[str, Any]
    ```
  - Validate all YAML configuration files against strict typed schemas prior to model instantiation.
- **Rationale**:
  - Enforces complete decoupling: changing an encoder or adding a GMM prior requires zero modifications to the training loop, CLI, or loss computation.
  - Catches typos and configuration errors at launch rather than during training.
- **Alternatives Considered**:
  - *Hardcoded if-else factory*: Brittle, requires editing central factory code whenever a new component is introduced.
  - *Tuple unpacking*: Fragile across disparate posterior/likelihood implementations.

---

## 6. Execution Runtime & CLI Strategy

- **Decision**: Unified local CLI application (`typer`) exposing `train`, `evaluate`, `generate`, `interpolate`, and `verify`, storing self-contained PyTorch checkpoints (`.pt`) containing weights, optimizer state, configuration schema, and metrics.
- **Rationale**:
  - The local workstation GPU (Quadro T2000 4GB VRAM) provides ample headroom to train CIFAR-10 at batch size 128 locally (~1 GB VRAM usage). A single unified CLI keeps the codebase minimal, structured, and free of redundant runtime wrappers.
- **Alternatives Considered**:
  - *Jupyter Notebooks*: Prone to hidden state, cell ordering issues, and lack of automated reproducibility.
  - *Monolithic scripts*: Lacks standardized CLI parameter parsing and structured artifact emission.
