# Graph Report - Hands-on VAE  (2026-09-26)

## Corpus Check
- 110 files · ~155,710 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 3 file(s) not represented in the graph (top: (none) 3)

## Summary
- 595 nodes · 1374 edges · 28 communities (23 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 57 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- CLI & Visualization Workflows
- Configuration Schema & Verification
- Core Mathematical Formulations & Latent Dynamics
- Convolutional Decoder & Group Normalization
- Evaluation & Benchmarking Pipeline (FID/IS)
- Specify Prerequisite Shell Utilities
- CIFAR-10 Data Pipeline
- Heteroscedastic Likelihood & Beta-NLL Subsystem
- Abstract Base Component Classes
- Variational Posterior & Reparameterization
- Backbone Implementations (CNN & Heteroscedastic)
- Checkpoint Serialization & Provenance
- Component Forward & Sampling Contracts
- ELBO Loss Objective & Decomposed Metrics
- Decorator Registries
- Registry Component Lookup & Resolution
- Training Lifecycle & Gradient Optimization
- Trainer Orchestration & Logging
- Seeding, Dataloaders & Device Execution
- Dynamic Model Factory (build_vae_from_config)
- Deliverables, Technical Reports & Documentation
- Likelihood Registration Decorators
- Architecture Deep Dive & Taxonomy Visuals
- Subclass Conformance Unit Tests
- Feature Specifications (001 & 002)
- CLI Package Initialization
- Active Latent Dimensionality Metrics
- Project Root Package

## God Nodes (most connected - your core abstractions)
1. `VAE` - 39 edges
2. `build_vae_from_config()` - 37 edges
3. `load_config()` - 31 edges
4. `device()` - 23 edges
5. `BaseLikelihood` - 21 edges
6. `validate_config()` - 20 edges
7. `BaseEncoder` - 19 edges
8. `unnormalize()` - 17 edges
9. `compute_fid_and_is()` - 17 edges
10. `interpolate_images()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Gaussian Mixture Model (GMM) Prior` --generalizes--> `StandardGaussianPrior`  [INFERRED]
  ARCHITECTURE_DEEP_DIVE.md → src/models/priors/standard_gaussian.py
- `Fréchet Inception Distance (FID)` --implemented_by--> `compute_fid_and_is()`  [EXTRACTED]
  docs/reports/001-baseline-vae-report.md → src/evaluation/metrics.py
- `Inception Score (IS)` --implemented_by--> `compute_fid_and_is()`  [EXTRACTED]
  docs/reports/001-baseline-vae-report.md → src/evaluation/metrics.py
- `Group Normalization (GroupNorm)` --used_in--> `CNNDecoder`  [EXTRACTED]
  ARCHITECTURE_DEEP_DIVE.md → src/models/decoders/cnn.py
- `Group Normalization (GroupNorm)` --used_in--> `CNNEncoder`  [EXTRACTED]
  ARCHITECTURE_DEEP_DIVE.md → src/models/encoders/cnn.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Generative Modeling Core Pipeline** — src_models_encoders_cnn_cnnencoder, src_models_posteriors_diagonal_diagonalgaussianposterior, src_models_priors_standard_gaussian_standardgaussianprior, src_models_decoders_cnn_cnndecoder, src_models_losses_elbo_elboloss [EXTRACTED 1.00]
- **Heteroscedastic Uncertainty Subsystem** — src_models_decoders_cnn_hetero_heteroscedasticcnndecoder, src_models_likelihoods_gaussian_hetero_heteroscedasticgaussianlikelihood, concept_heteroscedastic_likelihood, concept_beta_nll_stabilization [EXTRACTED 1.00]
- **GenCV003 Benchmarking Suite** — concept_frechet_inception_distance, concept_inception_score, concept_active_latent_units, src_evaluation_metrics_compute_fid_and_is, doc_gencv003_pdf [EXTRACTED 1.00]

## Communities (28 total, 5 thin omitted)

### Community 0 - "CLI & Visualization Workflows"
Cohesion: 0.05
Nodes (85): callback, command, math, matplotlib, matplotlib_pyplot, numpy, pil, sklearn_decomposition (+77 more)

### Community 1 - "Configuration Schema & Verification"
Cohesion: 0.06
Nodes (48): dataclasses, Execute Constitution Principle IV pre-training integrity verification., verify(), Configuration parsing and validation., ComponentConfig, ConfigError, DataSectionConfig, ExperimentConfig (+40 more)

### Community 2 - "Core Mathematical Formulations & Latent Dynamics"
Cohesion: 0.06
Nodes (28): Evidence Lower Bound (ELBO), Gaussian Mixture Model (GMM) Prior, Homoscedastic Gaussian Likelihood, L2 Conditional Mean Smoothing Trap, Spherical Prior Concentration, HomoscedasticGaussianLikelihood, Any, Tensor (+20 more)

### Community 3 - "Convolutional Decoder & Group Normalization"
Cohesion: 0.06
Nodes (27): Group Normalization (GroupNorm), Project Constitution v1.1.0, CNNDecoder, Tensor, 3-stage transposed convolutional decoder with GroupNorm and tanh bounding., Decode latent vector z into continuous observation parameters [-1, 1]. Args: z:…, CNNEncoder, Tensor (+19 more)

### Community 4 - "Evaluation & Benchmarking Pipeline (FID/IS)"
Cohesion: 0.07
Nodes (31): Fréchet Inception Distance (FID), Inception Score (IS), scipy, Quantitative model evaluation on test partitions., calculate_frechet_distance(), calculate_inception_score(), compute_fid_and_is(), InceptionFeatureExtractor (+23 more)

### Community 5 - "Specify Prerequisite Shell Utilities"
Cohesion: 0.13
Nodes (29): check-prerequisites.sh script, check_dir(), check_file(), find_specify_root(), format_speckit_command(), get_current_branch(), get_feature_paths(), get_invoke_separator() (+21 more)

### Community 6 - "CIFAR-10 Data Pipeline"
Cohesion: 0.08
Nodes (33): get_cifar10_dataloaders(), get_cifar10_datasets(), get_cifar10_transforms(), Any, Compose, DataLoader, Dataset, Path (+25 more)

### Community 7 - "Heteroscedastic Likelihood & Beta-NLL Subsystem"
Cohesion: 0.07
Nodes (26): Beta-NLL Loss Weighting, Heteroscedastic Gaussian Likelihood, fixture, pytest, HeteroscedasticCNNDecoder, Tensor, 3-stage transposed CNN decoder with dual heads predicting mean and pixel-wise…, Decode latent vector z into observation mean and pixel-wise log-variance map.… (+18 more)

### Community 8 - "Abstract Base Component Classes"
Cohesion: 0.16
Nodes (20): ABC, BaseDecoder, BaseEncoder, BaseLikelihood, BasePosterior, BasePosteriorDistribution, BasePrior, BasePriorDistribution (+12 more)

### Community 9 - "Variational Posterior & Reparameterization"
Cohesion: 0.12
Nodes (16): Full Correlated Covariance Posterior, Continuous Gaussian Reparameterization, DiagonalGaussianDistribution, DiagonalGaussianPosterior, Tensor, Diagonal Gaussian variational posterior instance q_phi(z|x)., Differentiable sampling via the reparameterization trick: z = mu + sigma * eps., Sample latent representation using the reparameterization trick. (+8 more)

### Community 10 - "Backbone Implementations (CNN & Heteroscedastic)"
Cohesion: 0.19
Nodes (13): Abstract Base Classes for modular VAE components., Heteroscedastic transposed convolutional neural network decoder., Convolutional neural network decoder backbone., Decoder architectures., Convolutional neural network encoder backbone., Heteroscedastic Gaussian observation likelihood with beta-NLL loss weighting., Homoscedastic Gaussian observation likelihood., Factorized diagonal Gaussian variational posterior distribution. (+5 more)

### Community 11 - "Checkpoint Serialization & Provenance"
Cohesion: 0.18
Nodes (17): datetime, Optimizer, load_checkpoint(), Any, Module, Path, Checkpoint serialization, restoration, and provenance tracking., Serialize model checkpoint with complete metadata and provenance. Args: path:… (+9 more)

### Community 12 - "Component Forward & Sampling Contracts"
Cohesion: 0.11
Nodes (11): Any, Tensor, Sample latent vector using reparameterization., Sample latent vectors from the prior., Forward pass. Returns feature tensor [B, hidden_dim]., Returns a PosteriorDistribution instance supporting .sample() and distribution…, Draws latent vectors directly from the prior [num_samples, latent_dim]., Computes or estimates D_KL(posterior || prior). Returns tensor [B]. (+3 more)

### Community 13 - "ELBO Loss Objective & Decomposed Metrics"
Cohesion: 0.17
Nodes (15): BaseLoss, Calculates scalar loss and decomposed metrics., Computes total training objective from VAEOutput and target observations., Evidence Lower Bound (ELBO) variational loss objective., Decorator to register a BaseLoss subclass., register_loss(), LossOutput, Type definitions and dataclasses for VAE inputs, outputs, and losses. (+7 more)

### Community 14 - "Decorator Registries"
Cohesion: 0.11
Nodes (14): Decorator to register a BaseEncoder subclass., Decorator to register a BaseDecoder subclass., Decorator to register a BasePosterior subclass., Decorator to register a BasePrior subclass., register_decoder(), register_encoder(), decorator(), register_posterior() (+6 more)

### Community 15 - "Registry Component Lookup & Resolution"
Cohesion: 0.18
Nodes (17): _ensure_default_components(), get_decoder(), get_encoder(), get_likelihood(), get_loss(), get_posterior(), get_prior(), Retrieve registered encoder by name. (+9 more)

### Community 16 - "Training Lifecycle & Gradient Optimization"
Cohesion: 0.15
Nodes (11): Typer CLI Application, Any, no_grad, Path, Execute a single training epoch., Execute validation pass., Load model, optimizer, scheduler state from checkpoint and return next epoch., Orchestrates model training, validation, metric aggregation, and checkpoint… (+3 more)

### Community 17 - "Trainer Orchestration & Logging"
Cohesion: 0.22
Nodes (9): json, Logger, pathlib, Training and validation execution loop for VAE models., get_logger(), Path, Structured logging utility with console and file handlers., Configures and returns a structured logger. Args: name: Name of the logger.… (+1 more)

### Community 18 - "Seeding, Dataloaders & Device Execution"
Cohesion: 0.20
Nodes (8): os, random, DataLoader, Module, Utility functions for seeding, device detection, and logging., Deterministic seeding utility for reproducible experiments across runtimes., Sets global random seeds for Python, NumPy, and PyTorch (CPU and CUDA).…, seed_everything()

### Community 19 - "Dynamic Model Factory (build_vae_from_config)"
Cohesion: 0.33
Nodes (7): build_vae_from_config(), _extract_params(), Any, Extract component parameters supporting flat dicts and nested 'params' dicts., Validate config schema, instantiate registered modules, and return composite…, Verify dynamic construction of composite VAE module from YAML config., test_build_vae_from_config()

### Community 20 - "Deliverables, Technical Reports & Documentation"
Cohesion: 0.40
Nodes (5): Baseline VAE Technical Report, Enhanced VAE Technical Report, GenCV003 Assignment Brief, Project Handoff State, Project Root README

### Community 21 - "Likelihood Registration Decorators"
Cohesion: 0.50
Nodes (3): Decorator to register a BaseLikelihood subclass., register_likelihood(), TLikelihood

### Community 22 - "Architecture Deep Dive & Taxonomy Visuals"
Cohesion: 0.67
Nodes (3): Architecture Visualization Diagram, Architecture Deep Dive, Architectural Enhancements Taxonomy

## Knowledge Gaps
- **16 isolated node(s):** `common.sh script`, `hands-on-vae`, `Evidence Lower Bound (ELBO)`, `Gaussian Mixture Model (GMM) Prior`, `Full Correlated Covariance Posterior` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 280 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VAE` connect `Convolutional Decoder & Group Normalization` to `CLI & Visualization Workflows`, `Core Mathematical Formulations & Latent Dynamics`, `Evaluation & Benchmarking Pipeline (FID/IS)`, `Heteroscedastic Likelihood & Beta-NLL Subsystem`, `Abstract Base Component Classes`, `Variational Posterior & Reparameterization`, `Backbone Implementations (CNN & Heteroscedastic)`, `ELBO Loss Objective & Decomposed Metrics`, `Training Lifecycle & Gradient Optimization`, `Dynamic Model Factory (build_vae_from_config)`?**
  _High betweenness centrality (0.154) - this node is a cross-community bridge._
- **Why does `build_vae_from_config()` connect `Dynamic Model Factory (build_vae_from_config)` to `CLI & Visualization Workflows`, `Configuration Schema & Verification`, `Convolutional Decoder & Group Normalization`, `Heteroscedastic Likelihood & Beta-NLL Subsystem`, `Abstract Base Component Classes`, `Backbone Implementations (CNN & Heteroscedastic)`, `Checkpoint Serialization & Provenance`, `Registry Component Lookup & Resolution`, `Trainer Orchestration & Logging`, `Seeding, Dataloaders & Device Execution`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `BaseLikelihood` connect `Abstract Base Component Classes` to `Core Mathematical Formulations & Latent Dynamics`, `Convolutional Decoder & Group Normalization`, `Heteroscedastic Likelihood & Beta-NLL Subsystem`, `Backbone Implementations (CNN & Heteroscedastic)`, `Component Forward & Sampling Contracts`, `ELBO Loss Objective & Decomposed Metrics`, `Registry Component Lookup & Resolution`, `Likelihood Registration Decorators`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `VAE` (e.g. with `extract_latent_embeddings()` and `plot_2d_latent_manifold()`) actually correct?**
  _`VAE` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `device()` (e.g. with `_load_tensor()` and `_load_tensor()`) actually correct?**
  _`device()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `common.sh script`, `hands-on-vae`, `Evidence Lower Bound (ELBO)` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `CLI & Visualization Workflows` be split into smaller, more focused modules?**
  _Cohesion score 0.050187265917602995 - nodes in this community are weakly interconnected._