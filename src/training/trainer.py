"""Training and validation execution loop for VAE models."""

import json
from pathlib import Path
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.registry import build_vae_from_config
from src.training.checkpoint import save_checkpoint
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything


class VAETrainer:
    """Orchestrates model training, validation, metric aggregation, and checkpoint serialization."""

    def __init__(
        self,
        config: dict[str, Any],
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config

        # Experiment settings
        exp_cfg = config.get("experiment", {})
        self.seed = exp_cfg.get("seed", 42)
        seed_everything(self.seed)

        # Device selection
        if device is not None:
            self.device = device
        else:
            dev_str = exp_cfg.get("device", "auto")
            if dev_str == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(dev_str)

        self.output_dir = Path(exp_cfg.get("output_dir", "artifacts/runs/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("vae.trainer", log_file=self.output_dir / "train.log")
        self.logger.info(f"Initializing VAE training on device: {self.device}")

        # Instantiate or assign model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = build_vae_from_config(config).to(self.device)

        # DataLoaders
        data_cfg = config.get("data", {})
        dataset_name = data_cfg.get("dataset", "cifar10").lower()
        if train_loader is not None and val_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            if dataset_name == "mnist":
                from src.data.mnist import get_mnist_dataloaders
                t_loader, v_loader, _ = get_mnist_dataloaders(
                    data_dir=data_cfg.get("data_dir", "data"),
                    batch_size=data_cfg.get("batch_size", 128),
                    val_split=data_cfg.get("val_split", 0.1),
                    num_workers=data_cfg.get("num_workers", 4),
                    seed=self.seed,
                )
            else:
                t_loader, v_loader, _ = get_cifar10_dataloaders(
                    data_dir=data_cfg.get("data_dir", "data"),
                    batch_size=data_cfg.get("batch_size", 128),
                    val_split=data_cfg.get("val_split", 0.1),
                    num_workers=data_cfg.get("num_workers", 4),
                    seed=self.seed,
                )
            self.train_loader = t_loader
            self.val_loader = v_loader

        # Training hyperparameters
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.lr = train_cfg.get("lr", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 1e-5)
        self.grad_clip = train_cfg.get("gradient_clip_val", 5.0)
        self.save_every = train_cfg.get("save_every", 5)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        sched_cfg = train_cfg.get("scheduler", {})
        if sched_cfg.get("type") == "cosine":
            min_lr = sched_cfg.get("min_lr", 1e-5)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=min_lr,
            )
        else:
            self.scheduler = None

        self.history: list[dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.global_step = 0

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute a single training epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Train]", leave=False)
        for batch in pbar:
            images = batch[0].to(self.device)
            self.optimizer.zero_grad()

            output = self.model(images)
            loss_out = self.model.compute_loss(output, images)

            loss_out.loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            batch_loss = loss_out.loss.item()
            batch_recon = loss_out.reconstruction_loss.item()
            batch_kl = loss_out.kl_divergence.item()

            total_loss += batch_loss
            total_recon += batch_recon
            total_kl += batch_kl

            self.global_step += 1
            pbar.set_postfix({
                "loss": f"{batch_loss:.2f}",
                "recon": f"{batch_recon:.2f}",
                "kl": f"{batch_kl:.2f}",
            })

        return {
            "train_loss": total_loss / num_batches,
            "train_recon": total_recon / num_batches,
            "train_kl": total_kl / num_batches,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Execute validation pass."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = len(self.val_loader)

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Val]", leave=False)
        for batch in pbar:
            images = batch[0].to(self.device)
            output = self.model(images)
            loss_out = self.model.compute_loss(output, images)

            total_loss += loss_out.loss.item()
            total_recon += loss_out.reconstruction_loss.item()
            total_kl += loss_out.kl_divergence.item()

        return {
            "val_loss": total_loss / num_batches,
            "val_recon": total_recon / num_batches,
            "val_kl": total_kl / num_batches,
        }

    def resume_from_checkpoint(self, path: Union[Path, str]) -> int:
        """Load model, optimizer, scheduler state from checkpoint and return next epoch."""
        from src.training.checkpoint import load_checkpoint
        ckpt = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict") and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("metrics", {}).get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        self.logger.info(f"Resumed from checkpoint {path} at epoch {start_epoch}")
        return start_epoch

    def train(self, start_epoch: int = 1) -> dict[str, Any]:
        """Run complete training lifecycle across all configured epochs."""
        self.logger.info(f"Starting training run from epoch {start_epoch} to {self.epochs}")

        for epoch in range(start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "global_step": self.global_step,
                "lr": self.optimizer.param_groups[0]["lr"],
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_record)

            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} (Recon: {train_metrics['train_recon']:.4f}, KL: {train_metrics['train_kl']:.4f}) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} (Recon: {val_metrics['val_recon']:.4f}, KL: {val_metrics['val_kl']:.4f})"
            )

            # Save latest checkpoint
            save_checkpoint(
                path=self.output_dir / "latest.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                global_step=self.global_step,
                config=self.config,
                metrics=epoch_record,
                seed=self.seed,
            )

            # Save best checkpoint
            val_loss = val_metrics["val_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                for name in ("best.pt", "best_checkpoint.pt"):
                    save_checkpoint(
                        path=self.output_dir / name,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        global_step=self.global_step,
                        config=self.config,
                        metrics=epoch_record,
                        seed=self.seed,
                    )
                self.logger.info(f"New best checkpoint saved (Val Loss: {val_loss:.4f})")

            # Save periodic checkpoint
            if epoch % self.save_every == 0:
                save_checkpoint(
                    path=self.output_dir / f"epoch_{epoch:03d}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    config=self.config,
                    metrics=epoch_record,
                    seed=self.seed,
                )

            # Persist metrics json
            with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

        # Save final checkpoint
        if self.history:
            save_checkpoint(
                path=self.output_dir / "final_checkpoint.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.epochs,
                global_step=self.global_step,
                config=self.config,
                metrics=self.history[-1],
                seed=self.seed,
            )

        # Render loss trajectory plot
        self._plot_loss_curves()

        self.logger.info("Training cycle complete.")
        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.epochs,
            "history": self.history,
            "best_checkpoint": str(self.output_dir / "best_checkpoint.pt"),
            "latest_checkpoint": str(self.output_dir / "latest.pt"),
        }

    def _plot_loss_curves(self) -> None:
        """Render and save loss trajectories to loss_curve.png."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not self.history:
                return

            epochs = [r["epoch"] for r in self.history]
            train_loss = [r["train_loss"] for r in self.history]
            val_loss = [r["val_loss"] for r in self.history]
            train_recon = [r["train_recon"] for r in self.history]
            val_recon = [r["val_recon"] for r in self.history]
            train_kl = [r["train_kl"] for r in self.history]
            val_kl = [r["val_kl"] for r in self.history]

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].plot(epochs, train_loss, label="Train ELBO")
            axes[0].plot(epochs, val_loss, label="Val ELBO")
            axes[0].set_title("Total ELBO Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(epochs, train_recon, label="Train Recon")
            axes[1].plot(epochs, val_recon, label="Val Recon")
            axes[1].set_title("Reconstruction Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(epochs, train_kl, label="Train KL")
            axes[2].plot(epochs, val_kl, label="Val KL")
            axes[2].set_title("KL Divergence")
            axes[2].set_xlabel("Epoch")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "loss_curve.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Could not render loss curve: {e}")
