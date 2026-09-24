"""Main CLI entrypoint for Hands-on VAE."""

import typer

app = typer.Typer(
    name="vae",
    help="Modular From-Scratch Variational Autoencoder CLI for CIFAR-10.",
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Hands-on VAE: Ground-up deep generative modeling and benchmarking."""
    pass


if __name__ == "__main__":
    app()
