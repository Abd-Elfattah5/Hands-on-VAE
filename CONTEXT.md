# Generative Modeling & Benchmarking Context

The domain of mathematical from-scratch generative vision modeling (Variational Autoencoders), deterministic benchmarking, and distribution evaluation.

## Language

### Core Generative Structures

**Encoder Backbone**:
Neural network component that maps an observed image to the parameter space of the variational latent distribution.
_Avoid_: Feature extractor, compression network

**Decoder Backbone**:
Neural network component that maps latent vectors into the parameter space of the observation likelihood distribution.
_Avoid_: Generator network, decompression network

**Variational Posterior ($q_\phi(z|x)$)**:
The approximate conditional probability distribution over the latent space parameterized by the encoder.
_Avoid_: Latent encoder, recognition model

**Latent Prior ($p(z)$)**:
The unconditioned base probability distribution over the latent space against which the variational posterior is regularized.
_Avoid_: Base distribution, target prior

**Observation Likelihood ($p_\theta(x|z)$)**:
The conditional probability distribution over observation space parameterized by the decoder output.
_Avoid_: Output distribution, reconstruction head

**Homoscedastic Gaussian Likelihood**:
A continuous Gaussian observation model assuming a constant, uniform noise variance across all pixels and samples.
_Avoid_: Simple MSE, standard L2 loss

**Group Normalization**:
A channel-group normalization method applied independently per sample, eliminating batch-size dependencies during generative sampling.
_Avoid_: Batch normalization, layer scaling

### Architecture & Extension Patterns

**Component Registry**:
A decoupled factory pattern binding string identifiers in configuration schemas to modular neural and mathematical primitives.
_Avoid_: Hardcoded switch, dynamic loader

**VAEOutput**:
A strongly-typed container encapsulating reconstructions, latent samples, posterior distributions, prior distributions, and auxiliary maps emitted from forward propagation.
_Avoid_: Output tuple, model prediction

### Mathematical Operations & Objectives

**Evidence Lower Bound (ELBO)**:
The variational objective decomposed into the expected log-likelihood (reconstruction) and the Kullback-Leibler divergence against the prior.
_Avoid_: Total loss, VAE cost

**Reparameterization**:
A differentiable stochastic sampling mechanism expressing latent random variables as a deterministic transformation of distribution parameters and independent noise.
_Avoid_: Sampling trick, noise injection

**Posterior Collapse**:
A degenerate training state where the variational posterior matches the prior across all inputs and the decoder ignores the latent variable.
_Avoid_: Mode collapse, KL vanishing

**Latent Traversal**:
Systematic exploration or linear/spherical interpolation across latent coordinates to inspect manifold smoothness and generative continuity.
_Avoid_: Latent walking, latent morphing
