"""Main CLI entrypoint for Hands-on VAE."""

import json
from pathlib import Path
from typing import Optional
import typer
import torch
import torch.nn as nn

from src.configs.schema import load_config, validate_config
from src.data.cifar10 import get_cifar10_dataloaders
from src.evaluation.evaluator import evaluate_model
from src.evaluation.latent_analysis import (
    extract_latent_embeddings,
    plot_2d_latent_manifold,
    plot_latent_pca,
    plot_latent_tsne,
)
from src.evaluation.metrics import compute_fid_and_is
from src.evaluation.visualizer import (
    generate_sample_grid,
    interpolate_2d_grid,
    interpolate_images,
    interpolate_synthetic_2d_grid,
    render_reconstruction_gallery,
    traverse_latent_dimensions,
)
from src.models.registry import build_vae_from_config
from src.training.checkpoint import restore_model_from_checkpoint
from src.training.trainer import VAETrainer
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

# Ensure all modular components are registered
import src.models.encoders.cnn
import src.models.decoders.cnn
import src.models.decoders.cnn_hetero
import src.models.posteriors.diagonal
import src.models.priors.standard_gaussian
import src.models.likelihoods.gaussian_homo
import src.models.likelihoods.gaussian_hetero
import src.models.losses.elbo

app = typer.Typer(
    name="vae",
    help="Modular From-Scratch Variational Autoencoder CLI for CIFAR-10.",
    add_completion=False,
)
logger = get_logger("vae.cli")


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    """Hands-on VAE: Ground-up deep generative modeling and benchmarking."""
    if version:
        typer.echo("Hands-on VAE version: 0.1.0")
        raise typer.Exit()


@app.command()
def verify(
    config: Path = typer.Option(
        ...,
        "-c",
        "--config",
        help="Path to experiment YAML configuration.",
        exists=True,
        readable=True,
    ),
) -> None:
    """Execute Constitution Principle IV pre-training integrity verification."""
    typer.echo(f"Verifying architecture integrity for: {config}")
    try:
        raw_cfg = load_config(config)
        validate_config(raw_cfg)
        model = build_vae_from_config(raw_cfg)
        model.train()

        in_channels = raw_cfg.get("model", {}).get("in_channels", 3)
        image_size = raw_cfg.get("model", {}).get("image_size", 32)
        latent_dim = raw_cfg.get("model", {}).get("latent_dim", 32)

        # 1. Forward Pass Shape Verification
        x = torch.randn(2, in_channels, image_size, image_size, requires_grad=True)
        output = model(x)

        if output.reconstruction.shape != (2, in_channels, image_size, image_size):
            raise AssertionError(
                f"Reconstruction shape mismatch: expected {(2, in_channels, image_size, image_size)}, "
                f"got {output.reconstruction.shape}"
            )
        if output.z.shape != (2, latent_dim):
            raise AssertionError(
                f"Latent z shape mismatch: expected {(2, latent_dim)}, got {output.z.shape}"
            )
        if output.reconstruction.min() < -1.0 or output.reconstruction.max() > 1.0:
            raise AssertionError(
                f"Reconstruction out of [-1, 1] bounds: "
                f"min={output.reconstruction.min().item():.4f}, max={output.reconstruction.max().item():.4f}"
            )
        typer.echo("✓ Shape contracts preserved through all network layers.")

        # 2. Gradient Flow Verification
        loss_output = model.compute_loss(output, x)
        loss_output.loss.backward()

        vanished_grads: list[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None or torch.all(param.grad == 0.0):
                    vanished_grads.append(name)

        if vanished_grads:
            raise AssertionError(f"Zero or vanished gradient flow detected in parameters: {vanished_grads}")
        typer.echo("✓ Differentiable reparameterization gradient flow verified with non-zero gradients.")

        # 3. Numerical Stability Verification
        if not torch.isfinite(loss_output.loss):
            raise AssertionError(f"Loss computation yielded non-finite value: {loss_output.loss.item()}")
        if not torch.isfinite(loss_output.reconstruction_loss):
            raise AssertionError("Reconstruction loss is not finite.")
        if not torch.isfinite(loss_output.kl_divergence):
            raise AssertionError("KL divergence is not finite.")
        typer.echo("✓ Numerical stability confirmed: finite loss and non-NaN gradients.")

        typer.secho("\nAll integrity checks passed successfully!", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"\nVerification failed: {e}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)


@app.command()
def train(
    config: Path = typer.Option(
        ...,
        "-c",
        "--config",
        help="Path to experiment YAML configuration.",
        exists=True,
        readable=True,
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "-r",
        "--resume",
        help="Path to checkpoint to resume training from.",
        exists=True,
        readable=True,
    ),
    seed: Optional[int] = typer.Option(
        None,
        "-s",
        "--seed",
        help="Override experiment random seed.",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Override number of training epochs.",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Override batch size.",
    ),
) -> None:
    """Train a VAE model according to declarative YAML configuration."""
    raw_cfg = load_config(config)
    validate_config(raw_cfg)

    if seed is not None:
        raw_cfg.setdefault("experiment", {})["seed"] = seed
    if epochs is not None:
        raw_cfg.setdefault("training", {})["epochs"] = epochs
    if batch_size is not None:
        raw_cfg.setdefault("data", {})["batch_size"] = batch_size

    typer.echo(f"Starting VAE training using configuration: {config}")
    trainer = VAETrainer(config=raw_cfg)

    start_epoch = 1
    if resume is not None:
        typer.echo(f"Resuming training from checkpoint: {resume}")
        start_epoch = trainer.resume_from_checkpoint(resume)

    results = trainer.train(start_epoch=start_epoch)
    typer.secho(
        f"Training complete! Best validation loss: {results['best_val_loss']:.4f}\n"
        f"Checkpoints saved in: {trainer.output_dir}",
        fg=typer.colors.GREEN,
        bold=True,
    )


def _get_dataloaders_from_config(
    config: dict,
    data_dir: Optional[Path] = None,
    batch_size: int = 128,
):
    dataset_name = config.get("data", {}).get("dataset", "cifar10").lower()
    target_data_dir = str(data_dir) if data_dir else config.get("data", {}).get("data_dir", "data")
    seed = config.get("experiment", {}).get("seed", 42)

    if dataset_name == "mnist":
        from src.data.mnist import get_mnist_dataloaders
        return get_mnist_dataloaders(data_dir=target_data_dir, batch_size=batch_size, seed=seed)
    else:
        from src.data.cifar10 import get_cifar10_dataloaders
        return get_cifar10_dataloaders(data_dir=target_data_dir, batch_size=batch_size, seed=seed)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(
        ...,
        "-k",
        "--checkpoint",
        help="Path to .pt checkpoint file.",
        exists=True,
        readable=True,
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "-d",
        "--data-dir",
        help="Override dataset directory.",
    ),
    batch_size: int = typer.Option(
        128,
        "-b",
        "--batch-size",
        help="Evaluation batch size.",
    ),
) -> None:
    """Evaluate a saved model checkpoint on test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_dict = restore_model_from_checkpoint(checkpoint, map_location=device)
    config = ckpt_dict.get("config", {})

    _, _, test_loader = _get_dataloaders_from_config(config, data_dir=data_dir, batch_size=batch_size)

    metrics = evaluate_model(model=model, data_loader=test_loader, device=device)
    latent_dim = config.get("model", {}).get("latent_dim", 32)

    typer.echo("\nEvaluation Results:")
    typer.echo(f"├── Test ELBO: {metrics['test_loss']:.2f}")
    typer.echo(f"├── Reconstruction Loss (MSE): {metrics['test_mse']:.4f}")
    typer.echo(f"├── KL Divergence: {metrics['test_kl']:.2f}")
    typer.echo(f"└── Active Units (Var(q) > 0.01): {metrics['active_units']}/{latent_dim}")

    # Emit eval_metrics.json in checkpoint directory
    ckpt_dir = Path(checkpoint).parent
    out_file = ckpt_dir / "eval_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    typer.echo(f"\nSaved evaluation metrics to: {out_file}")


@app.command()
def generate(
    checkpoint: Path = typer.Option(
        ...,
        "-k",
        "--checkpoint",
        help="Path to .pt checkpoint file.",
        exists=True,
        readable=True,
    ),
    num_samples: int = typer.Option(
        64,
        "-n",
        "--num-samples",
        help="Number of samples to generate (square number recommended).",
    ),
    out: Path = typer.Option(
        Path("artifacts/samples/sample_grid.png"),
        "-o",
        "--out",
        help="Output image filepath.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "-s",
        "--seed",
        help="Random seed for deterministic sampling.",
    ),
    upscale: int = typer.Option(
        4,
        "--upscale",
        help="Tile upscaling factor (4x converts 32x32 to 128x128 per tile, yielding crisp 1024x1024 canvases).",
    ),
) -> None:
    """Generate synthetic images by sampling latent vectors from prior."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = restore_model_from_checkpoint(checkpoint, map_location=device)

    out_path = generate_sample_grid(
        model=model,
        num_samples=num_samples,
        device=device,
        out_path=out,
        seed=seed,
        upscale=upscale,
    )
    typer.secho(f"Successfully generated {num_samples} samples: {out_path}", fg=typer.colors.GREEN)


@app.command()
def interpolate(
    checkpoint: Path = typer.Option(
        ...,
        "-k",
        "--checkpoint",
        help="Path to .pt checkpoint file.",
        exists=True,
        readable=True,
    ),
    img1: Optional[Path] = typer.Option(
        None,
        "--img1",
        help="Path to first image (or top-left image for 2D grid).",
    ),
    img2: Optional[Path] = typer.Option(
        None,
        "--img2",
        help="Path to second image (or top-right image for 2D grid).",
    ),
    img3: Optional[Path] = typer.Option(
        None,
        "--img3",
        help="Path to third image (bottom-left image for 4-corner 2D grid).",
    ),
    img4: Optional[Path] = typer.Option(
        None,
        "--img4",
        help="Path to fourth image (bottom-right image for 4-corner 2D grid).",
    ),
    steps: int = typer.Option(
        8,
        "--steps",
        help="Number of transition steps (or grid dimension for 2D grid).",
    ),
    out: Path = typer.Option(
        Path("artifacts/interpolations/strip.png"),
        "-o",
        "--out",
        help="Output image filepath.",
    ),
    synthetic: bool = typer.Option(
        False,
        "--synthetic",
        help="Draw 4 independent anchor vectors directly from prior N(0, I) without real images.",
    ),
    upscale: int = typer.Option(
        4,
        "--upscale",
        help="Tile upscaling factor (default 4x for high-resolution rendering).",
    ),
) -> None:
    """Interpolate in latent space (supports 2-image 1D strip, 4-image 2D grid, or pure synthetic prior 2D grid)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = restore_model_from_checkpoint(checkpoint, map_location=device)

    if synthetic:
        out_path = interpolate_synthetic_2d_grid(
            model=model,
            grid_size=steps,
            out_path=out,
            device=device,
            upscale=upscale,
        )
        typer.secho(
            f"Successfully rendered pure synthetic 2D prior interpolation grid ({steps}x{steps}): {out_path}",
            fg=typer.colors.GREEN,
        )
    elif img3 is not None and img4 is not None:
        if img1 is None or img2 is None:
            typer.secho("Error: 4-corner 2D grid requires --img1, --img2, --img3, and --img4", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        out_path = interpolate_2d_grid(
            model=model,
            img_tl=img1,
            img_tr=img2,
            img_bl=img3,
            img_br=img4,
            grid_size=steps,
            out_path=out,
            device=device,
            upscale=upscale,
        )
        typer.secho(
            f"Successfully rendered 4-corner 2D interpolation grid ({steps}x{steps}): {out_path}",
            fg=typer.colors.GREEN,
        )
    else:
        if img1 is None or img2 is None:
            typer.secho("Error: 1D interpolation requires --img1 and --img2 (or use --synthetic)", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        out_path = interpolate_images(
            model=model,
            img1=img1,
            img2=img2,
            steps=steps,
            out_path=out,
            device=device,
            upscale=upscale,
        )
        typer.secho(
            f"Successfully rendered interpolation strip: {out_path}",
            fg=typer.colors.GREEN,
        )


@app.command()
def benchmark(
    checkpoint: Path = typer.Option(
        ...,
        "-k",
        "--checkpoint",
        help="Path to .pt checkpoint file.",
        exists=True,
        readable=True,
    ),
    num_samples: int = typer.Option(
        5000,
        "-n",
        "--num-samples",
        help="Number of samples to evaluate for FID and Inception Score.",
    ),
    batch_size: int = typer.Option(
        64,
        "-b",
        "--batch-size",
        help="Batch size for Inception feature extraction.",
    ),
    out: Path = typer.Option(
        Path("artifacts/eval/benchmark_metrics.json"),
        "-o",
        "--out",
        help="Output benchmark JSON filepath.",
    ),
) -> None:
    """Execute quantitative benchmarking (ELBO, Reconstruction MSE, KL, Active Units, FID, IS)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Loading checkpoint {checkpoint} on {device}...")
    model, ckpt_dict = restore_model_from_checkpoint(checkpoint, map_location=device)
    config = ckpt_dict.get("config", {})

    _, _, test_loader = _get_dataloaders_from_config(config, batch_size=batch_size)

    typer.echo("Step 1/2: Evaluating test split ELBO, reconstruction error, and active latent units...")
    eval_metrics = evaluate_model(model=model, data_loader=test_loader, device=device)

    typer.echo(f"Step 2/2: Computing Fréchet Inception Distance (FID) and Inception Score (IS) over {num_samples} samples...")
    gen_metrics = compute_fid_and_is(
        model=model,
        real_loader=test_loader,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
    )

    combined_metrics = {
        **eval_metrics,
        **gen_metrics,
        "checkpoint": str(checkpoint),
        "latent_dim": getattr(model, "latent_dim", model.prior.latent_dim),
    }

    out_file = Path(out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)

    typer.secho("\n================ Quantitative Benchmark Results ================", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"├── Test ELBO:                  {combined_metrics['test_loss']:.2f}")
    typer.echo(f"├── Reconstruction Loss (MSE):  {combined_metrics['test_mse']:.4f}")
    typer.echo(f"├── KL Divergence:              {combined_metrics['test_kl']:.2f}")
    typer.echo(f"├── Active Latent Units:        {combined_metrics['active_units']}/{combined_metrics['latent_dim']}")
    typer.echo(f"├── Fréchet Inception Distance: {combined_metrics['fid']:.2f}")
    typer.echo(f"└── Inception Score (IS):       {combined_metrics['inception_score_mean']:.2f} ± {combined_metrics['inception_score_std']:.2f}")
    typer.secho(f"================================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho(f"Metrics persisted to: {out_file}", fg=typer.colors.GREEN)


@app.command("plot-latent")
def plot_latent(
    checkpoint: Path = typer.Option(
        ...,
        "-k",
        "--checkpoint",
        help="Path to .pt checkpoint file.",
        exists=True,
        readable=True,
    ),
    num_samples: int = typer.Option(
        2500,
        "-n",
        "--num-samples",
        help="Number of test samples to embed in t-SNE and PCA.",
    ),
    dim_x: int = typer.Option(
        0,
        "--dim-x",
        help="First latent dimension coordinate for 2D manifold traversal.",
    ),
    dim_y: int = typer.Option(
        1,
        "--dim-y",
        help="Second latent dimension coordinate for 2D manifold traversal.",
    ),
    grid_size: int = typer.Option(
        10,
        "--grid-size",
        help="Meshgrid dimension for 2D manifold traversal.",
    ),
    out_dir: Path = typer.Option(
        Path("artifacts/eval"),
        "-o",
        "--out-dir",
        help="Output directory for generated plots.",
    ),
    upscale: int = typer.Option(
        4,
        "--upscale",
        help="Canvas tile upscaling factor for high-res output.",
    ),
) -> None:
    """Generate comprehensive latent space visual diagnostics (t-SNE, PCA, 2D manifold, sweeps, gallery)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_dict = restore_model_from_checkpoint(checkpoint, map_location=device)
    config = ckpt_dict.get("config", {})

    _, _, test_loader = _get_dataloaders_from_config(config, batch_size=128)

    dataset_name = config.get("data", {}).get("dataset", "cifar10").lower()
    classes = [f"Digit {i}" for i in range(10)] if dataset_name == "mnist" else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Extracting {num_samples} latent embeddings for t-SNE & PCA...")
    mu_arr, labels_arr = extract_latent_embeddings(model=model, data_loader=test_loader, device=device, max_samples=num_samples)

    typer.echo("Generating t-SNE scatter plot...")
    tsne_path = plot_latent_tsne(mu_arr, labels_arr, out_path=out_dir / "latent_tsne.png", class_names=classes)

    typer.echo("Generating PCA scatter plot...")
    pca_path = plot_latent_pca(mu_arr, labels_arr, out_path=out_dir / "latent_pca.png", class_names=classes)

    typer.echo(f"Generating 2D manifold traversal grid ({grid_size}x{grid_size}) along dimensions ({dim_x}, {dim_y})...")
    manifold_path = plot_2d_latent_manifold(
        model=model,
        dim_x=dim_x,
        dim_y=dim_y,
        grid_size=grid_size,
        out_path=out_dir / "latent_manifold_2d.png",
        device=device,
        upscale=upscale,
    )

    typer.echo("Generating 1D latent coordinate sweeps...")
    sweep_path = traverse_latent_dimensions(
        model=model,
        dimensions=(0, 1, 2, 3, 4, 5, 6, 7),
        steps=10,
        out_path=out_dir / "latent_traversal_1d.png",
        device=device,
        upscale=upscale,
    )

    typer.echo("Generating side-by-side reconstruction gallery...")
    first_batch = next(iter(test_loader))[0][:8]  # 8 images
    gallery_path, gallery_metrics = render_reconstruction_gallery(
        model=model,
        images=first_batch,
        out_path=out_dir / "reconstruction_gallery.png",
        device=device,
        upscale=upscale,
    )

    typer.secho("\nAll latent diagnostic plots generated successfully:", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"├── t-SNE Scatter:              {tsne_path}")
    typer.echo(f"├── PCA Scatter:                {pca_path}")
    typer.echo(f"├── 2D Manifold Grid:           {manifold_path}")
    typer.echo(f"├── 1D Axis Sweeps:             {sweep_path}")
    typer.echo(f"└── Reconstruction Gallery:     {gallery_path} (MSE: {gallery_metrics['gallery_mse']:.4f}, PSNR: {gallery_metrics['gallery_psnr']:.1f} dB)")


if __name__ == "__main__":
    app()
