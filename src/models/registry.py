"""Component registries and model factory for modular VAE architecture."""

from typing import Any, Callable, TypeVar

from src.models.base import (
    BaseDecoder,
    BaseEncoder,
    BaseLikelihood,
    BaseLoss,
    BasePosterior,
    BasePrior,
)

TEncoder = TypeVar("TEncoder", bound=type[BaseEncoder])
TDecoder = TypeVar("TDecoder", bound=type[BaseDecoder])
TPosterior = TypeVar("TPosterior", bound=type[BasePosterior])
TPrior = TypeVar("TPrior", bound=type[BasePrior])
TLikelihood = TypeVar("TLikelihood", bound=type[BaseLikelihood])
TLoss = TypeVar("TLoss", bound=type[BaseLoss])

ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {}
DECODER_REGISTRY: dict[str, type[BaseDecoder]] = {}
POSTERIOR_REGISTRY: dict[str, type[BasePosterior]] = {}
PRIOR_REGISTRY: dict[str, type[BasePrior]] = {}
LIKELIHOOD_REGISTRY: dict[str, type[BaseLikelihood]] = {}
LOSS_REGISTRY: dict[str, type[BaseLoss]] = {}


def register_encoder(name: str) -> Callable[[TEncoder], TEncoder]:
    """Decorator to register a BaseEncoder subclass."""
    def decorator(cls: TEncoder) -> TEncoder:
        if not issubclass(cls, BaseEncoder):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseEncoder")
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def register_decoder(name: str) -> Callable[[TDecoder], TDecoder]:
    """Decorator to register a BaseDecoder subclass."""
    def decorator(cls: TDecoder) -> TDecoder:
        if not issubclass(cls, BaseDecoder):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseDecoder")
        DECODER_REGISTRY[name] = cls
        return cls
    return decorator


def register_posterior(name: str) -> Callable[[TPosterior], TPosterior]:
    """Decorator to register a BasePosterior subclass."""
    def decorator(cls: TPosterior) -> TPosterior:
        if not issubclass(cls, BasePosterior):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BasePosterior")
        POSTERIOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_prior(name: str) -> Callable[[TPrior], TPrior]:
    """Decorator to register a BasePrior subclass."""
    def decorator(cls: TPrior) -> TPrior:
        if not issubclass(cls, BasePrior):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BasePrior")
        PRIOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_likelihood(name: str) -> Callable[[TLikelihood], TLikelihood]:
    """Decorator to register a BaseLikelihood subclass."""
    def decorator(cls: TLikelihood) -> TLikelihood:
        if not issubclass(cls, BaseLikelihood):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseLikelihood")
        LIKELIHOOD_REGISTRY[name] = cls
        return cls
    return decorator


def register_loss(name: str) -> Callable[[TLoss], TLoss]:
    """Decorator to register a BaseLoss subclass."""
    def decorator(cls: TLoss) -> TLoss:
        if not issubclass(cls, BaseLoss):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseLoss")
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_default_components() -> None:
    """Ensure baseline components are imported so decorators are registered."""
    try:
        import src.models.encoders.cnn  # noqa: F401
        import src.models.decoders.cnn  # noqa: F401
        import src.models.posteriors.diagonal  # noqa: F401
        import src.models.priors.standard_gaussian  # noqa: F401
        import src.models.likelihoods.gaussian_homo  # noqa: F401
        import src.models.losses.elbo  # noqa: F401
    except ImportError:
        pass


def get_encoder(name: str) -> type[BaseEncoder]:
    """Retrieve registered encoder by name."""
    if name not in ENCODER_REGISTRY:
        _ensure_default_components()
    if name not in ENCODER_REGISTRY:
        raise KeyError(
            f"Encoder '{name}' not found. Available encoders: {sorted(list(ENCODER_REGISTRY.keys()))}"
        )
    return ENCODER_REGISTRY[name]


def get_decoder(name: str) -> type[BaseDecoder]:
    """Retrieve registered decoder by name."""
    if name not in DECODER_REGISTRY:
        _ensure_default_components()
    if name not in DECODER_REGISTRY:
        raise KeyError(
            f"Decoder '{name}' not found. Available decoders: {sorted(list(DECODER_REGISTRY.keys()))}"
        )
    return DECODER_REGISTRY[name]


def get_posterior(name: str) -> type[BasePosterior]:
    """Retrieve registered posterior by name."""
    if name not in POSTERIOR_REGISTRY:
        _ensure_default_components()
    if name not in POSTERIOR_REGISTRY:
        raise KeyError(
            f"Posterior '{name}' not found. Available posteriors: {sorted(list(POSTERIOR_REGISTRY.keys()))}"
        )
    return POSTERIOR_REGISTRY[name]


def get_prior(name: str) -> type[BasePrior]:
    """Retrieve registered prior by name."""
    if name not in PRIOR_REGISTRY:
        _ensure_default_components()
    if name not in PRIOR_REGISTRY:
        raise KeyError(
            f"Prior '{name}' not found. Available priors: {sorted(list(PRIOR_REGISTRY.keys()))}"
        )
    return PRIOR_REGISTRY[name]


def get_likelihood(name: str) -> type[BaseLikelihood]:
    """Retrieve registered likelihood by name."""
    if name not in LIKELIHOOD_REGISTRY:
        _ensure_default_components()
    if name not in LIKELIHOOD_REGISTRY:
        raise KeyError(
            f"Likelihood '{name}' not found. Available likelihoods: {sorted(list(LIKELIHOOD_REGISTRY.keys()))}"
        )
    return LIKELIHOOD_REGISTRY[name]


def get_loss(name: str) -> type[BaseLoss]:
    """Retrieve registered loss by name."""
    if name not in LOSS_REGISTRY:
        _ensure_default_components()
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Loss '{name}' not found. Available losses: {sorted(list(LOSS_REGISTRY.keys()))}"
        )
    return LOSS_REGISTRY[name]


def _extract_params(comp_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract component parameters supporting flat dicts and nested 'params' dicts."""
    if "params" in comp_cfg and isinstance(comp_cfg["params"], dict):
        return dict(comp_cfg["params"])
    return {k: v for k, v in comp_cfg.items() if k != "name"}


def build_vae_from_config(config: dict[str, Any]) -> Any:
    """Validate config schema, instantiate registered modules, and return composite VAE."""
    from src.configs.schema import validate_config

    validate_config(config)

    model_cfg = config.get("model", config)
    latent_dim = model_cfg["latent_dim"]
    in_channels = model_cfg.get("in_channels", 3)
    image_size = model_cfg.get("image_size", 32)

    # 1. Encoder
    enc_cfg = model_cfg["encoder"]
    enc_cls = get_encoder(enc_cfg["name"])
    enc_params = _extract_params(enc_cfg)
    if "in_channels" not in enc_params:
        enc_params["in_channels"] = in_channels
    if "image_size" not in enc_params:
        enc_params["image_size"] = image_size
    encoder = enc_cls(**enc_params)

    # 2. Posterior
    post_cfg = model_cfg["posterior"]
    post_cls = get_posterior(post_cfg["name"])
    post_params = _extract_params(post_cfg)
    if "in_features" not in post_params:
        post_params["in_features"] = encoder.out_features
    if "latent_dim" not in post_params:
        post_params["latent_dim"] = latent_dim
    posterior = post_cls(**post_params)

    # 3. Prior
    prior_cfg = model_cfg["prior"]
    prior_cls = get_prior(prior_cfg["name"])
    prior_params = _extract_params(prior_cfg)
    if "latent_dim" not in prior_params:
        prior_params["latent_dim"] = latent_dim
    prior = prior_cls(**prior_params)

    # 4. Decoder
    dec_cfg = model_cfg["decoder"]
    dec_cls = get_decoder(dec_cfg["name"])
    dec_params = _extract_params(dec_cfg)
    if "latent_dim" not in dec_params:
        dec_params["latent_dim"] = latent_dim
    if "out_channels" not in dec_params:
        dec_params["out_channels"] = in_channels
    if "image_size" not in dec_params:
        dec_params["image_size"] = image_size
    decoder = dec_cls(**dec_params)

    # 5. Likelihood
    like_cfg = model_cfg["likelihood"]
    like_cls = get_likelihood(like_cfg["name"])
    like_params = _extract_params(like_cfg)
    likelihood = like_cls(**like_params)

    # 6. Loss
    loss_cfg = model_cfg["loss"]
    loss_cls = get_loss(loss_cfg["name"])
    loss_params = _extract_params(loss_cfg)
    if "likelihood" not in loss_params:
        loss_params["likelihood"] = likelihood
    loss_fn = loss_cls(**loss_params)

    from src.models.vae import VAE
    return VAE(
        encoder=encoder,
        posterior=posterior,
        prior=prior,
        decoder=decoder,
        likelihood=likelihood,
        loss_fn=loss_fn,
        config=config,
    )
