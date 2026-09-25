"""Sample synthesis, latent space interpolation, and visual grid rendering."""

import math
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import torch
import torchvision.utils as vutils

from src.data.cifar10 import get_cifar10_transforms, unnormalize
from src.models.vae import VAE


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation (SLERP) between two vectors."""
    low_norm = low / (torch.norm(low, dim=-1, keepdim=True) + 1e-8)
    high_norm = high / (torch.norm(high, dim=-1, keepdim=True) + 1e-8)
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    # Fall back to linear interpolation if angle is negligible
    if torch.all(torch.abs(so) < 1e-6):
        return (1.0 - val) * low + val * high

    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def lerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between two vectors."""
    return (1.0 - val) * low + val * high


def generate_sample_grid(
    model: VAE,
    num_samples: int = 64,
    device: Optional[torch.device] = None,
    out_path: Union[str, Path] = "artifacts/samples/sample_grid.png",
    seed: Optional[int] = None,
) -> Path:
    """Generate visual sample grid by sampling latent vectors from prior N(0, I).

    Args:
        model: Trained VAE model instance.
        num_samples: Number of samples to draw (e.g. 64 for 8x8 grid).
        device: Target execution device.
        out_path: Filepath where formatted grid PNG will be saved.
        seed: Optional seed for deterministic generation.

    Returns:
        Path to the saved image grid artifact.
    """
    if device is None:
        device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, device=device)
        unnorm_samples = unnormalize(samples)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    grid_nrow = int(math.isqrt(num_samples))
    if grid_nrow * grid_nrow != num_samples:
        grid_nrow = 8

    vutils.save_image(
        unnorm_samples,
        str(out_file),
        nrow=grid_nrow,
        normalize=False,
        padding=2,
    )

    return out_file


def interpolate_images(
    model: VAE,
    img1: Union[str, Path, torch.Tensor],
    img2: Union[str, Path, torch.Tensor],
    steps: int = 10,
    mode: str = "linear",
    out_path: Union[str, Path] = "artifacts/interpolations/strip.png",
    device: Optional[torch.device] = None,
) -> Path:
    """Encode two input images, interpolate latent path, decode, and save transition strip.

    Args:
        model: Trained VAE model instance.
        img1: Path to first image file or tensor [1, C, H, W] in [-1, 1].
        img2: Path to second image file or tensor [1, C, H, W] in [-1, 1].
        steps: Number of transition steps across the interpolation trajectory.
        mode: Interpolation method ('linear' or 'spherical').
        out_path: Destination path for rendered transition strip.
        device: Target execution device.

    Returns:
        Path to the saved interpolation strip artifact.
    """
    if device is None:
        device = next(model.parameters()).device

    transform = get_cifar10_transforms()

    def _load_tensor(img_input: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        if isinstance(img_input, (str, Path)):
            pil_img = Image.open(str(img_input)).convert("RGB")
            # Resize if necessary to model image size (32x32)
            pil_img = pil_img.resize((32, 32))
            tensor = transform(pil_img).unsqueeze(0)
        else:
            tensor = img_input
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        return tensor.to(device)

    t1 = _load_tensor(img1)
    t2 = _load_tensor(img2)

    model.eval()
    with torch.no_grad():
        # Encode deterministically to posterior mean
        z1 = model.encode(t1, deterministic=True)
        z2 = model.encode(t2, deterministic=True)

        frames: list[torch.Tensor] = []
        for i in range(steps):
            alpha = float(i) / float(steps - 1) if steps > 1 else 0.0
            if mode == "spherical":
                z_interp = slerp(alpha, z1, z2)
            else:
                z_interp = lerp(alpha, z1, z2)
            decoded = model.decode(z_interp)
            frames.append(unnormalize(decoded))

        strip = torch.cat(frames, dim=0)  # [steps, C, H, W]

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        strip,
        str(out_file),
        nrow=steps,
        normalize=False,
        padding=2,
    )

    return out_file


def interpolate_2d_grid(
    model: VAE,
    img_tl: Union[str, Path, torch.Tensor],
    img_tr: Union[str, Path, torch.Tensor],
    img_bl: Union[str, Path, torch.Tensor],
    img_br: Union[str, Path, torch.Tensor],
    grid_size: int = 8,
    out_path: Union[str, Path] = "artifacts/interpolations/grid_2d.png",
    device: Optional[torch.device] = None,
) -> Path:
    """Perform 2D bilinear latent interpolation across 4 corner anchor images.

    Interpolation formula for coordinates (u, v) in [0, 1] x [0, 1]:
        z(u, v) = (1 - u)(1 - v) z_tl + u(1 - v) z_tr + (1 - u)v z_bl + uv z_br

    Args:
        model: Trained VAE model instance.
        img_tl: Top-left anchor image (path or tensor).
        img_tr: Top-right anchor image (path or tensor).
        img_bl: Bottom-left anchor image (path or tensor).
        img_br: Bottom-right anchor image (path or tensor).
        grid_size: Number of steps along each dimension (total grid = grid_size x grid_size).
        out_path: Filepath where the output 2D image grid will be saved.
        device: Target execution device.

    Returns:
        Path to the saved 2D interpolation grid artifact.
    """
    if device is None:
        device = next(model.parameters()).device

    transform = get_cifar10_transforms()

    def _load_tensor(img_input: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        if isinstance(img_input, (str, Path)):
            pil_img = Image.open(str(img_input)).convert("RGB")
            pil_img = pil_img.resize((32, 32))
            tensor = transform(pil_img).unsqueeze(0)
        else:
            tensor = img_input
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        return tensor.to(device)

    t_tl = _load_tensor(img_tl)
    t_tr = _load_tensor(img_tr)
    t_bl = _load_tensor(img_bl)
    t_br = _load_tensor(img_br)

    model.eval()
    with torch.no_grad():
        z_tl = model.encode(t_tl, deterministic=True)
        z_tr = model.encode(t_tr, deterministic=True)
        z_bl = model.encode(t_bl, deterministic=True)
        z_br = model.encode(t_br, deterministic=True)

        frames: list[torch.Tensor] = []
        for row in range(grid_size):
            v = float(row) / float(grid_size - 1) if grid_size > 1 else 0.0
            for col in range(grid_size):
                u = float(col) / float(grid_size - 1) if grid_size > 1 else 0.0
                z_uv = (
                    (1.0 - u) * (1.0 - v) * z_tl
                    + u * (1.0 - v) * z_tr
                    + (1.0 - u) * v * z_bl
                    + u * v * z_br
                )
                decoded = model.decode(z_uv)
                frames.append(unnormalize(decoded))

        grid = torch.cat(frames, dim=0)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        grid,
        str(out_file),
        nrow=grid_size,
        normalize=False,
        padding=2,
    )

    return out_file
