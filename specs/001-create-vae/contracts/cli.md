# CLI Interface Contract: `vae`

**Feature**: `001-create-vae`
**Status**: Completed

The command-line interface provides the primary entry point for all operational workflows (Constitution Principle V). Built using `typer`.

---

## 1. Global Invariant & Options

```bash
vae [OPTIONS] COMMAND [ARGS]...
```

- `--help`: Shows help message and exits.
- `--version`: Displays version information (`0.1.0`).

---

## 2. Commands

### 2.1 `train`

Trains a VAE model according to a YAML experiment configuration.

```bash
vae train --config <path_to_yaml> [--resume <checkpoint_path>] [--seed <int>]
```

- **Options**:
  - `-c, --config PATH`: Path to experiment YAML configuration file. **[Required]**
  - `-r, --resume PATH`: Path to a previous `.pt` checkpoint to resume training from. *[Optional]*
  - `-s, --seed INTEGER`: Overrides the random seed specified in the config. *[Optional]*
- **Exit Codes**:
  - `0`: Success, training completed and final checkpoint saved.
  - `1`: Configuration validation error.
  - `2`: Runtime error (e.g., CUDA OOM, NaN loss encountered).
- **Artifacts Emitted**:
  - `<output_dir>/best_checkpoint.pt`: Checkpoint with lowest validation loss.
  - `<output_dir>/final_checkpoint.pt`: Checkpoint from final epoch.
  - `<output_dir>/metrics.json`: Per-epoch train/val loss, reconstruction error, and KL divergence.
  - `<output_dir>/loss_curve.png`: Rendered plot of training and validation trajectories.

---

### 2.2 `evaluate`

Evaluates a saved checkpoint on the dataset test split.

```bash
vae evaluate --checkpoint <checkpoint_path> [--data-dir <path>] [--batch-size <int>]
```

- **Options**:
  - `-k, --checkpoint PATH`: Path to `.pt` checkpoint file. **[Required]**
  - `-d, --data-dir PATH`: Override data directory. *[Default: from checkpoint config]*
  - `-b, --batch-size INTEGER`: Evaluation batch size. *[Default: 128]*
- **Output**:
  - Prints structured evaluation metrics to console:
    ```text
    Evaluation Results:
    ├── Test ELBO: 1243.52
    ├── Reconstruction Loss (MSE): 0.0142
    ├── KL Divergence: 48.31
    └── Active Units (Var(q) > 0.01): 28/32
    ```
  - Emits `<checkpoint_dir>/eval_metrics.json`.

---

### 2.3 `generate`

Samples latent vectors from the prior and synthesizes visual images.

```bash
vae generate --checkpoint <checkpoint_path> [--num-samples <int>] [--out <output_path>] [--seed <int>]
```

- **Options**:
  - `-k, --checkpoint PATH`: Path to `.pt` checkpoint file. **[Required]**
  - `-n, --num-samples INTEGER`: Number of samples to generate (must be a square number for grid, e.g. 16, 64). *[Default: 64]*
  - `-o, --out PATH`: Path where the output PNG/JPG grid will be saved. *[Default: artifacts/samples/sample_grid.png]*
  - `-s, --seed INTEGER`: Random seed for latent prior sampling. *[Default: None]*
- **Output**:
  - Saves an un-normalized image grid $[0, 1]$ visualizing synthesized samples.

---

### 2.4 `interpolate`

Encodes two images and renders an intermediate latent interpolation strip.

```bash
vae interpolate --checkpoint <checkpoint_path> --img1 <path> --img2 <path> [--steps <int>] [--out <path>]
```

- **Options**:
  - `-k, --checkpoint PATH`: Path to `.pt` checkpoint file. **[Required]**
  - `--img1 PATH`: Path to first image. **[Required]**
  - `--img2 PATH`: Path to second image. **[Required]**
  - `--steps INTEGER`: Number of discrete transition frames. *[Default: 10]*
  - `-o, --out PATH`: Output image strip path. *[Default: artifacts/interpolations/strip.png]*

---

### 2.5 `verify`

Executes the Constitution Principle IV pre-training integrity check on the configured architecture.

```bash
vae verify --config <path_to_yaml>
```

- **Options**:
  - `-c, --config PATH`: Path to experiment YAML configuration. **[Required]**
- **Checks Performed**:
  1. Synthetic forward pass with batch size 2 checking tensor shapes through all layers.
  2. Differentiable reparameterization gradient check (ensures non-zero gradients on all encoder/decoder weights).
  3. Numerical stability check verifying that loss computation does not produce NaN or Inf.
- **Exit Codes**:
  - `0`: All integrity assertions passed.
  - `1`: Assertion failure (prints layer shape mismatch or vanishing gradient diagnostics).
