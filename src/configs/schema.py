"""Configuration schema and validation for VAE experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


class ConfigError(ValueError):
    """Raised when configuration schema or dimensional contract validation fails."""
    pass


@dataclass
class ComponentConfig:
    """Configuration for a modular registered component."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSectionConfig:
    """Experiment metadata and execution environment."""
    name: str = "cifar10_baseline"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "artifacts/runs/cifar10_baseline"


@dataclass
class DataSectionConfig:
    """Dataset loading and preprocessing configuration."""
    dataset: str = "cifar10"
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 4
    val_split: float = 0.1


@dataclass
class ModelSectionConfig:
    """Model architecture configuration and subcomponent specifications."""
    latent_dim: int = 32
    in_channels: int = 3
    image_size: int = 32
    encoder: dict[str, Any] = field(default_factory=dict)
    posterior: dict[str, Any] = field(default_factory=dict)
    prior: dict[str, Any] = field(default_factory=dict)
    decoder: dict[str, Any] = field(default_factory=dict)
    likelihood: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSectionConfig:
    """Optimizer, scheduler, and training loop parameters."""
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 1.0e-5
    scheduler: dict[str, Any] = field(default_factory=dict)
    gradient_clip_val: float = 5.0
    save_every: int = 5
    eval_every: int = 1


@dataclass
class ExperimentConfig:
    """Top-level unified experiment configuration."""
    experiment: ExperimentSectionConfig
    data: DataSectionConfig
    model: ModelSectionConfig
    training: TrainingSectionConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file from disk."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Expected YAML file at {config_path} to contain a mapping, got {type(data).__name__}")
    return data


def validate_model_config(model_cfg: dict[str, Any]) -> None:
    """Validate model sub-dictionary against dimensional and architectural constraints."""
    # Check required subcomponents
    required_keys = ["latent_dim", "encoder", "posterior", "prior", "decoder", "likelihood", "loss"]
    for k in required_keys:
        if k not in model_cfg:
            raise ConfigError(f"Model configuration missing required key: '{k}'")

    latent_dim = model_cfg["latent_dim"]
    if not isinstance(latent_dim, int) or latent_dim < 1:
        raise ConfigError(f"latent_dim must be a positive integer, got: {latent_dim}")

    in_channels = model_cfg.get("in_channels", 3)
    if not isinstance(in_channels, int) or in_channels < 1:
        raise ConfigError(f"in_channels must be a positive integer, got: {in_channels}")

    image_size = model_cfg.get("image_size", 32)
    if not isinstance(image_size, int) or image_size < 4:
        raise ConfigError(f"image_size must be an integer >= 4, got: {image_size}")

    # Validate Encoder
    enc = model_cfg["encoder"]
    if not isinstance(enc, dict) or "name" not in enc:
        raise ConfigError("encoder configuration must be a mapping with a 'name' key")
    if enc["name"] == "cnn":
        channels = enc.get("channels", [32, 64, 128])
        if not isinstance(channels, list) or len(channels) == 0:
            raise ConfigError("CNN encoder channels must be a non-empty list of integers")
        num_downsamples = len(channels)
        downsample_factor = 2 ** num_downsamples
        if image_size % downsample_factor != 0:
            raise ConfigError(
                f"image_size ({image_size}) must be divisible by 2^{num_downsamples} = {downsample_factor} "
                f"for CNN encoder downsampling stages"
            )
        spatial_dim = image_size // downsample_factor
        if spatial_dim < 1:
            raise ConfigError(f"Spatial dimension collapsed to {spatial_dim} after downsampling")
        num_groups = enc.get("num_groups", 8)
        for i, c in enumerate(channels):
            if c % num_groups != 0:
                raise ConfigError(
                    f"CNN encoder channel {c} at stage {i} is not divisible by num_groups={num_groups}"
                )

    # Validate Posterior
    post = model_cfg["posterior"]
    if not isinstance(post, dict) or "name" not in post:
        raise ConfigError("posterior configuration must be a mapping with a 'name' key")
    if post["name"] == "diagonal":
        clamp_min = post.get("clamp_min", -20.0)
        clamp_max = post.get("clamp_max", 2.0)
        if clamp_min >= clamp_max:
            raise ConfigError(f"clamp_min ({clamp_min}) must be strictly less than clamp_max ({clamp_max})")

    # Validate Prior
    prior = model_cfg["prior"]
    if not isinstance(prior, dict) or "name" not in prior:
        raise ConfigError("prior configuration must be a mapping with a 'name' key")

    # Validate Decoder
    dec = model_cfg["decoder"]
    if not isinstance(dec, dict) or "name" not in dec:
        raise ConfigError("decoder configuration must be a mapping with a 'name' key")
    if dec["name"] == "cnn":
        channels = dec.get("channels", [128, 64, 32])
        if not isinstance(channels, list) or len(channels) == 0:
            raise ConfigError("CNN decoder channels must be a non-empty list of integers")
        num_groups = dec.get("num_groups", 8)
        for i, c in enumerate(channels):
            if c % num_groups != 0:
                raise ConfigError(
                    f"CNN decoder channel {c} at stage {i} is not divisible by num_groups={num_groups}"
                )

    # Validate Likelihood
    like = model_cfg["likelihood"]
    if not isinstance(like, dict) or "name" not in like:
        raise ConfigError("likelihood configuration must be a mapping with a 'name' key")
    if like["name"] == "gaussian_homo":
        fixed_sigma = like.get("fixed_sigma", 1.0)
        if fixed_sigma <= 0:
            raise ConfigError(f"gaussian_homo fixed_sigma must be positive, got: {fixed_sigma}")

    # Validate Loss
    loss = model_cfg["loss"]
    if not isinstance(loss, dict) or "name" not in loss:
        raise ConfigError("loss configuration must be a mapping with a 'name' key")
    if "beta" in loss and loss["beta"] < 0:
        raise ConfigError(f"loss beta must be non-negative, got: {loss['beta']}")


def validate_config(config: dict[str, Any]) -> ExperimentConfig:
    """Validate full or partial configuration dictionary and return typed ExperimentConfig."""
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a dictionary, got: {type(config).__name__}")

    # Allow model-only configuration validation
    if "model" not in config:
        if "latent_dim" in config and "encoder" in config:
            validate_model_config(config)
            return ExperimentConfig(
                experiment=ExperimentSectionConfig(),
                data=DataSectionConfig(),
                model=ModelSectionConfig(
                    latent_dim=config["latent_dim"],
                    in_channels=config.get("in_channels", 3),
                    image_size=config.get("image_size", 32),
                    encoder=config["encoder"],
                    posterior=config["posterior"],
                    prior=config["prior"],
                    decoder=config["decoder"],
                    likelihood=config["likelihood"],
                    loss=config["loss"],
                ),
                training=TrainingSectionConfig(),
            )
        else:
            raise ConfigError("Configuration missing required top-level 'model' section")

    # Validate top-level sections
    exp_dict = config.get("experiment", {})
    data_dict = config.get("data", {})
    model_dict = config.get("model", {})
    training_dict = config.get("training", {})

    # Data validation
    if "val_split" in data_dict:
        val_split = data_dict["val_split"]
        if not (0.0 < val_split < 1.0):
            raise ConfigError(f"data.val_split must be in range (0.0, 1.0), got: {val_split}")

    if "batch_size" in data_dict:
        batch_size = data_dict["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigError(f"data.batch_size must be a positive integer, got: {batch_size}")

    # Model validation
    validate_model_config(model_dict)

    # Training validation
    if "epochs" in training_dict:
        epochs = training_dict["epochs"]
        if not isinstance(epochs, int) or epochs < 1:
            raise ConfigError(f"training.epochs must be an integer >= 1, got: {epochs}")

    if "lr" in training_dict:
        lr = training_dict["lr"]
        if lr <= 0:
            raise ConfigError(f"training.lr must be positive, got: {lr}")

    return ExperimentConfig(
        experiment=ExperimentSectionConfig(**exp_dict) if exp_dict else ExperimentSectionConfig(),
        data=DataSectionConfig(**data_dict) if data_dict else DataSectionConfig(),
        model=ModelSectionConfig(
            latent_dim=model_dict["latent_dim"],
            in_channels=model_dict.get("in_channels", 3),
            image_size=model_dict.get("image_size", 32),
            encoder=model_dict["encoder"],
            posterior=model_dict["posterior"],
            prior=model_dict["prior"],
            decoder=model_dict["decoder"],
            likelihood=model_dict["likelihood"],
            loss=model_dict["loss"],
        ),
        training=TrainingSectionConfig(**training_dict) if training_dict else TrainingSectionConfig(),
    )
