# Hands-on Variational Autoencoder (VAE)

A modular, from-scratch implementation and benchmarking pipeline for Variational Autoencoders in PyTorch.

## Overview

This project implements deep generative Variational Autoencoder architectures from foundational mathematical and tensor primitives, supporting rigorous quantitative evaluation (ELBO, reconstruction loss, FID, Inception Score) and qualitative latent diagnostics on public benchmark datasets (CIFAR-10).

## Project Structure

```text
├── CONTEXT.md         # Domain modeling glossary
├── configs/            # Declarative YAML experiment configurations
├── docs/               # Architecture Decision Records (ADRs)
│   └── adr/
├── specs/              # Feature specifications and validation checklists
│   └── 001-create-vae/
├── src/                # Modular VAE codebase
│   ├── cli/            # Unified CLI application
│   ├── data/           # Dataset ingestion and preprocessing pipelines
│   ├── models/         # Encoder, decoder, posterior, prior, and likelihood registries
│   ├── training/       # Training loops, loss objectives, and checkpoint management
│   ├── evaluation/     # Metrics (FID, IS) and visual sampling
│   └── utils/          # Deterministic seeding and logging utilities
└── tests/              # Test-driven tensor shape and gradient integrity tests
```

## Features

- **From-Scratch Mathematical Rigor**: Explicit encoder-decoder topology, Gaussian reparameterization trick, and Evidence Lower Bound (ELBO) loss decomposition.
- **Pluggable Component Registry**: Extensible interfaces for encoder backbones, posterior formulations, latent priors, decoder backbones, likelihood models, and loss objectives.
- **Reproducible Pipeline**: Strict deterministic random seeding across runtimes with versioned checkpoint provenance.
- **Dual Execution Modes**: Unified local CLI for workstation workflows and companion Jupyter notebooks for cloud acceleration (e.g., Google Colab).
