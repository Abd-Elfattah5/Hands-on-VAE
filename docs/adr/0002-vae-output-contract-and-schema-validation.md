# 0002. VAEOutput Contract & Typed Schema Validation

To decouple neural feature extraction from loss computation across all 7 component categories, we standardize model forward passes to return a typed `VAEOutput` container holding the reconstruction tensor, latent sample vector $z$, posterior distribution object $q_\phi(z|x)$, prior distribution object $p(z)$, and an auxiliary dictionary. Furthermore, all declarative YAML configurations are validated against a strict schema prior to instantiation to prevent shape mismatches and silent failures before GPU memory allocation.
