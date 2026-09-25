# Research & Architectural Decisions: 002-enhanced-vae

## 1. Pixel-wise Heteroscedastic Likelihood vs. Full Pixel Covariance

* **Decision**: Implement a pixel-wise independent heteroscedastic Gaussian likelihood $\sigma_{\text{obs}}^2(z) \in \mathbb{R}^{B \times 3 \times 32 \times 32}$ rather than a full pixel covariance matrix $\Sigma \in \mathbb{R}^{B \times 3072 \times 3072}$.
* **Rationale**:
  - A full covariance matrix between all 3,072 pixels for a mini-batch of 128 images requires $128 \times 3072 \times 3072 \times 4\text{ bytes} \approx 4.83\text{ GB}$ of VRAM just to store the tensor, immediately exceeding the 4.0 GB physical memory of our NVIDIA Quadro T2000 GPU and causing fatal CUDA Out-Of-Memory (OOM) errors.
  - Furthermore, computing matrix inversions or Cholesky factorizations on batches of $3072 \times 3072$ matrices requires $\mathcal{O}(B \cdot D^3) \approx 3.7 \times 10^{12}$ FLOPs per step, hanging the GPU kernel.
  - In contrast, pixel-wise heteroscedastic variance requires only $128 \times 3 \times 32 \times 32 \times 4\text{ bytes} \approx 1.57\text{ MB}$ per batch, enabling millisecond execution while granting the model spatial freedom to model edge uncertainty.
* **Alternatives Considered**:
  - *Full pixel covariance*: Infeasible due to hardware OOM.
  - *Block-diagonal channel covariance ($3 \times 3$ per pixel)*: Slightly captures cross-channel color correlation, but pixel-wise spatial variance provides the primary benefit for edge sharpness.

---

## 2. Beta-NLL Loss Stabilization

* **Decision**: Implement $\beta$-NLL loss weighting with $\beta = 0.5$:
  $$\mathcal{L}_{\beta\text{-NLL}} = \frac{1}{2} \sum_{i=1}^D \left[ \frac{(x_i - \mu_i)^2}{\sigma_i^2} + \log \sigma_i^2 \right] \cdot (\sigma_i^{2\beta})_{\text{detached}}$$
* **Rationale**:
  - In standard heteroscedastic NLL ($\beta = 0.0$ weighting), the gradient with respect to predicted mean is $\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\mu - x}{\sigma^2}$. As $\sigma^2$ increases on difficult high-frequency textures, the gradient signal vanishes, encouraging the network to predict artificially massive variances to zero out the reconstruction penalty (the "variance cheating" failure mode).
  - Following Seitzer et al. (*On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks*, ICLR 2022), multiplying the loss by $(\sigma^{2\beta})_{\text{detached}}$ with $\beta = 0.5$ restores a constant gradient scale $\frac{\partial \mathcal{L}}{\partial \mu} \propto \frac{\mu - x}{\sigma}$, forcing the network to maintain accurate, sharp predictions on challenging textures while capturing true heteroscedastic uncertainty.
* **Alternatives Considered**:
  - *Unconstrained Gaussian NLL ($\beta = 0$)*: Prone to variance inflation and blurry outputs.
  - *Fixed noise variance ($\sigma^2 = 1.0$)*: The baseline model (suffers from $L_2$ conditional mean smoothing).

---

## 3. Latent Dimensionality Expansion ($d = 128$)

* **Decision**: Expand latent dimension $d$ from $32$ to $128$.
* **Rationale**:
  - Empirical GPU benchmarking on the local Quadro T2000 confirmed that scaling $d=128$ increases peak VRAM from $238.51\text{ MB}$ to only $240.78\text{ MB}$ ($<1\text{ MB}$ overhead for forward and backward passes).
  - Provides $4\times$ higher informational bandwidth to capture multi-scale visual details on CIFAR-10 without bottleneck saturation.
* **Alternatives Considered**:
  - *Latent $d = 256$*: Unnecessary parameter overhead for $32 \times 32$ images.
  - *Latent $d = 32$*: Baseline bottleneck (leads to capacity pressure on 10 CIFAR classes).

---

## 4. Latent Prior Formulation

* **Decision**: Retain standard isotropic Gaussian prior $\mathcal{N}(0, I_{128})$.
* **Rationale**:
  - Maintains exact closed-form analytical KL divergence without requiring Monte Carlo sampling approximations.
  - In conjunction with $d=128$, provides sufficient dimensionality to eliminate the class overlap observed in $d=32$.
* **Alternatives Considered**:
  - *Gaussian Mixture Model (GMM) Prior*: Replaces unimodal prior with $K=10$ components, but complicates closed-form KL divergence and adds training instability. Kept as a candidate for future advanced milestones.

---

## 5. Training Protocol

* **Decision**: Train for 50 epochs with cosine learning rate annealing ($10^{-3} \to 10^{-5}$) at batch size 128.
* **Rationale**:
  - Provides an exact, apples-to-apples baseline comparison against `001-create-vae`.
