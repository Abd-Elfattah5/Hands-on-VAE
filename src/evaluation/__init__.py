"""Evaluation and visualization routines."""

from src.evaluation.evaluator import evaluate_model
from src.evaluation.latent_analysis import (
    extract_latent_embeddings,
    plot_2d_latent_manifold,
    plot_latent_pca,
    plot_latent_tsne,
)
from src.evaluation.metrics import (
    calculate_frechet_distance,
    calculate_inception_score,
    compute_fid_and_is,
)
from src.evaluation.visualizer import (
    generate_sample_grid,
    interpolate_2d_grid,
    interpolate_images,
    interpolate_synthetic_2d_grid,
    lerp,
    render_reconstruction_gallery,
    slerp,
    traverse_latent_dimensions,
    upscale_tensor,
)

__all__ = [
    "evaluate_model",
    "compute_fid_and_is",
    "calculate_frechet_distance",
    "calculate_inception_score",
    "extract_latent_embeddings",
    "plot_latent_tsne",
    "plot_latent_pca",
    "plot_2d_latent_manifold",
    "generate_sample_grid",
    "interpolate_images",
    "interpolate_2d_grid",
    "interpolate_synthetic_2d_grid",
    "traverse_latent_dimensions",
    "render_reconstruction_gallery",
    "upscale_tensor",
    "slerp",
    "lerp",
]
