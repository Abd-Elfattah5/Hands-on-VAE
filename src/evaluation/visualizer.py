"""Sample synthesis, latent space interpolation, and visual grid rendering."""

import math
from pathlib import Path
from typing import Optional, Sequence, Union
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from src.data.cifar10 import get_cifar10_transforms, unnormalize
from src.models.vae import VAE


def upscale_tensor(tensor: torch.Tensor, factor: int = 4) -> torch.Tensor:
    """Upscale image tensor using bicubic interpolation for high-resolution canvas display.

    Args:
        tensor: Image tensor of shape [N, C, H, W] in [0, 1].
        factor: Upscaling multiplier (e.g., 4 converts 32x32 to 128x128).

    Returns:
        Upscaled tensor of shape [N, C, H * factor, W * factor] clamped to [0, 1].
    """
    if factor <= 1:
        return tensor
    return F.interpolate(
        tensor,
        scale_factor=float(factor),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)


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
    upscale: int = 4,
) -> Path:
    """Generate visual sample grid by sampling latent vectors from prior N(0, I).

    Args:
        model: Trained VAE model instance.
        num_samples: Number of samples to draw (e.g. 64 for 8x8 grid).
        device: Target execution device.
        out_path: Filepath where formatted grid PNG will be saved.
        seed: Optional seed for deterministic generation.
        upscale: Upscaling factor for high-resolution canvas output (default 4x = 1024x1024).

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
        if upscale > 1:
            unnorm_samples = upscale_tensor(unnorm_samples, factor=upscale)

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
        padding=2 * upscale,
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
    upscale: int = 4,
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
        upscale: Tile upscaling multiplier.

    Returns:
        Path to the saved interpolation strip artifact.
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

    t1 = _load_tensor(img1)
    t2 = _load_tensor(img2)

    model.eval()
    with torch.no_grad():
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

        strip = torch.cat(frames, dim=0)
        if upscale > 1:
            strip = upscale_tensor(strip, factor=upscale)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        strip,
        str(out_file),
        nrow=steps,
        normalize=False,
        padding=2 * upscale,
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
    upscale: int = 4,
) -> Path:
    """Perform 2D bilinear latent interpolation across 4 corner anchor images."""
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
        if upscale > 1:
            grid = upscale_tensor(grid, factor=upscale)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        grid,
        str(out_file),
        nrow=grid_size,
        normalize=False,
        padding=2 * upscale,
    )

    return out_file


def interpolate_synthetic_2d_grid(
    model: VAE,
    grid_size: int = 8,
    out_path: Union[str, Path] = "artifacts/interpolations/synthetic_grid_2d.png",
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    upscale: int = 4,
) -> Path:
    """Pure synthetic 2D bilinear interpolation from prior N(0, I) without real image encoding.

    Draws 4 independent anchor vectors directly from the latent prior:
        z_tl, z_tr, z_bl, z_br ~ N(0, I)
    and interpolates smoothly across the 2D plane.

    Args:
        model: Trained VAE model instance.
        grid_size: Number of steps along each axis.
        out_path: Filepath where the output grid PNG will be saved.
        device: Target execution device.
        seed: Optional random seed for reproducible prior sampling.
        upscale: Tile upscaling multiplier (default 4x).

    Returns:
        Path to the saved synthetic 2D interpolation grid.
    """
    if device is None:
        device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        anchors = model.prior.sample(4, device=device)
        z_tl, z_tr, z_bl, z_br = anchors[0:1], anchors[1:2], anchors[2:3], anchors[3:4]

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
        if upscale > 1:
            grid = upscale_tensor(grid, factor=upscale)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        grid,
        str(out_file),
        nrow=grid_size,
        normalize=False,
        padding=2 * upscale,
    )

    return out_file


def traverse_latent_dimensions(
    model: VAE,
    dimensions: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
    steps: int = 10,
    range_limit: float = 3.0,
    out_path: Union[str, Path] = "artifacts/interpolations/latent_traversal.png",
    device: Optional[torch.device] = None,
    upscale: int = 4,
) -> Path:
    """Perform 1D coordinate sweeps along individual latent dimensions.

    For each selected dimension j:
        z_j is varied smoothly from -range_limit to +range_limit,
        while all other latent dimensions remain fixed at 0.0.

    Args:
        model: Trained VAE model instance.
        dimensions: Sequence of latent coordinate indices to sweep.
        steps: Number of points along each dimension sweep.
        range_limit: Coordinate boundary [-range_limit, +range_limit] (default 3.0 = 3 sigma).
        out_path: Filepath where traversal strip grid will be saved.
        device: Target execution device.
        upscale: Tile upscaling multiplier.

    Returns:
        Path to the saved traversal artifact.
    """
    if device is None:
        device = next(model.parameters()).device

    latent_dim = getattr(model, "latent_dim", None)
    if latent_dim is None:
        latent_dim = model.prior.latent_dim

    model.eval()
    with torch.no_grad():
        sweep_vals = torch.linspace(-range_limit, range_limit, steps, device=device)
        frames: list[torch.Tensor] = []

        for dim_idx in dimensions:
            z_batch = torch.zeros(steps, latent_dim, device=device)
            z_batch[:, dim_idx] = sweep_vals
            decoded = model.decode(z_batch)
            frames.append(unnormalize(decoded))

        grid = torch.cat(frames, dim=0)  # [len(dimensions) * steps, C, H, W]
        if upscale > 1:
            grid = upscale_tensor(grid, factor=upscale)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        grid,
        str(out_file),
        nrow=steps,
        normalize=False,
        padding=2 * upscale,
    )

    return out_file


def render_reconstruction_gallery(
    model: VAE,
    images: torch.Tensor,
    out_path: Union[str, Path] = "artifacts/eval/reconstruction_gallery.png",
    device: Optional[torch.device] = None,
    upscale: int = 4,
) -> tuple[Path, dict[str, float]]:
    """Render side-by-side original vs. reconstructed image pairs with quantitative PSNR and MSE.

    Args:
        model: Trained VAE model instance.
        images: Batch of real test images [N, C, H, W] in [-1, 1].
        out_path: Output filepath for the gallery PNG.
        device: Target execution device.
        upscale: Tile upscaling multiplier.

    Returns:
        Tuple of (saved_file_path, metrics_summary).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    images = images.to(device)

    with torch.no_grad():
        output = model(images)
        reconstruction = output.reconstruction

        orig_unnorm = unnormalize(images)
        recon_unnorm = unnormalize(reconstruction)

        # Compute per-sample and mean MSE / PSNR on [0, 1] range
        mse = F.mse_loss(recon_unnorm, orig_unnorm).item()
        psnr = 10.0 * math.log10(1.0 / (mse + 1e-8))

        # Interleave originals and reconstructions: [orig_0, recon_0, orig_1, recon_1, ...]
        pairs: list[torch.Tensor] = []
        for i in range(images.size(0)):
            pairs.append(orig_unnorm[i : i + 1])
            pairs.append(recon_unnorm[i : i + 1])

        gallery = torch.cat(pairs, dim=0)
        if upscale > 1:
            gallery = upscale_tensor(gallery, factor=upscale)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        gallery,
        str(out_file),
        nrow=4,  # 2 pairs per row
        normalize=False,
        padding=2 * upscale,
    )

    return out_file, {"gallery_mse": mse, "gallery_psnr": psnr}
