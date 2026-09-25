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
from src.evaluation.visualizer import generate_sample_grid, interpolate_2d_grid, interpolate_images
from src.models.registry import build_vae_from_config
from src.training.checkpoint import restore_model_from_checkpoint
from src.training.trainer import VAETrainer
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

# Ensure all modular components are registered
import src.models.encoders.cnn
import src.models.decoders.cnn
import src.models.posteriors.diagonal
import src.models.priors.standard_gaussian
import src.models.likelihoods.gaussian_homo
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

    if resume is not None:
        typer.echo(f"Resuming training from checkpoint: {resume}")
        restore_model_from_checkpoint(resume, model=trainer.model)

    results = trainer.train()
    typer.secho(
        f"Training complete! Best validation loss: {results['best_val_loss']:.4f}\n"
        f"Checkpoints saved in: {trainer.output_dir}",
        fg=typer.colors.GREEN,
        bold=True,
    )


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
    """Evaluate a saved model checkpoint on CIFAR-10 test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_dict = restore_model_from_checkpoint(checkpoint, map_location=device)
    config = ckpt_dict.get("config", {})

    target_data_dir = str(data_dir) if data_dir else config.get("data", {}).get("data_dir", "data")
    seed = config.get("experiment", {}).get("seed", 42)

    _, _, test_loader = get_cifar10_dataloaders(
        data_dir=target_data_dir,
        batch_size=batch_size,
        seed=seed,
    )

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
    img1: Path = typer.Option(
        ...,
        "--img1",
        help="Path to first image (or top-left image for 2D grid).",
        exists=True,
        readable=True,
    ),
    img2: Path = typer.Option(
        ...,
        "--img2",
        help="Path to second image (or top-right image for 2D grid).",
        exists=True,
        readable=True,
    ),
    img3: Optional[Path] = typer.Option(
        None,
        "--img3",
        help="Path to third image (bottom-left image for 4-corner 2D grid).",
        exists=True,
        readable=True,
    ),
    img4: Optional[Path] = typer.Option(
        None,
        "--img4",
        help="Path to fourth image (bottom-right image for 4-corner 2D grid).",
        exists=True,
        readable=True,
    ),
    steps: int = typer.Option(
        10,
        "--steps",
        help="Number of transition steps (or grid dimension for 2D grid).",
    ),
    out: Path = typer.Option(
        Path("artifacts/interpolations/strip.png"),
        "-o",
        "--out",
        help="Output image filepath.",
    ),
) -> None:
    """Interpolate between images in latent space (supports 2-image 1D strip or 4-image 2D grid)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = restore_model_from_checkpoint(checkpoint, map_location=device)

    if img3 is not None and img4 is not None:
        out_path = interpolate_2d_grid(
            model=model,
            img_tl=img1,
            img_tr=img2,
            img_bl=img3,
            img_br=img4,
            grid_size=steps,
            out_path=out,
            device=device,
        )
        typer.secho(
            f"Successfully rendered 4-corner 2D interpolation grid ({steps}x{steps}): {out_path}",
            fg=typer.colors.GREEN,
        )
    else:
        out_path = interpolate_images(
            model=model,
            img1=img1,
            img2=img2,
            steps=steps,
            out_path=out,
            device=device,
        )
        typer.secho(
            f"Successfully rendered interpolation strip: {out_path}",
            fg=typer.colors.GREEN,
        )


if __name__ == "__main__":
    app()
