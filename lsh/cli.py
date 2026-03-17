"""
Command-line interface for LSH-DP.

Usage:
    lshdp run --config config.yaml
    lshdp run --config config.yaml --bin-width 0.01 --device cuda
    lshdp validate config.yaml
    lshdp info
    lshdp cite
    lshdp init-config
"""

from __future__ import annotations

import sys
from typing import Optional

import click

from lsh import __version__, __citation__
from lsh.banner import print_banner, print_citation
from lsh.config import PipelineConfig, load_config, validate_config, save_example_config
from lsh.logging_utils import setup_logging, log_hardware_info, log_config_summary


@click.group()
@click.version_option(version=__version__, prog_name="lshdp")
def cli() -> None:
    """LSH-DP: Locality-Sensitive Hashing Dataset Reduction for Deep Potential Training."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML configuration file.")
@click.option("--bin-width", type=float, default=None, help="Override hashing bin width.")
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default=None, help="Override compute device.")
@click.option("--input-file", type=str, default=None, help="Override input trajectory file.")
@click.option("--output-dir", type=str, default=None, help="Override output directory.")
@click.option("--start-step", type=click.IntRange(1, 7), default=None, help="Start from this pipeline step.")
@click.option("--end-step", type=click.IntRange(1, 7), default=None, help="End at this pipeline step.")
@click.option("--n-jobs", type=int, default=None, help="Parallel workers for SOAP (-1 = all cores).")
def run(
    config: str,
    bin_width: Optional[float],
    device: Optional[str],
    input_file: Optional[str],
    output_dir: Optional[str],
    start_step: Optional[int],
    end_step: Optional[int],
    n_jobs: Optional[int],
) -> None:
    """Run the LSH-DP pipeline."""
    # Build overrides dict
    overrides: dict = {}
    if bin_width is not None:
        overrides.setdefault("hashing", {})["bin_width"] = bin_width
    if device is not None:
        overrides["device"] = device
    if input_file is not None:
        overrides.setdefault("io", {})["input_file"] = input_file
    if output_dir is not None:
        overrides.setdefault("io", {})["output_dir"] = output_dir
    if start_step is not None:
        overrides["start_step"] = start_step
    if end_step is not None:
        overrides["end_step"] = end_step
    if n_jobs is not None:
        overrides.setdefault("soap", {})["n_jobs"] = n_jobs

    cfg = load_config(config, overrides)

    # Validate
    issues = validate_config(cfg)
    if issues:
        click.echo("Configuration errors:", err=True)
        for issue in issues:
            click.echo(f"  ✗ {issue}", err=True)
        sys.exit(1)

    # Banner
    print_banner(cfg)

    # Logging
    logger = setup_logging(cfg.io.output_dir, log_file=cfg.log_file)
    log_hardware_info(logger, cfg.device)
    log_config_summary(logger, cfg)

    # Run
    from lsh.pipeline import run_pipeline
    run_pipeline(cfg)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str) -> None:
    """Validate a YAML configuration file."""
    cfg = load_config(config_path)
    issues = validate_config(cfg)
    if issues:
        click.echo("Validation FAILED:")
        for issue in issues:
            click.echo(f"  ✗ {issue}")
        sys.exit(1)
    else:
        click.echo("✓ Configuration is valid.")


@cli.command()
def info() -> None:
    """Display package information and environment details."""
    import platform
    import torch

    click.echo(f"LSH-DP v{__version__}")
    click.echo(f"Python   : {platform.python_version()}")
    click.echo(f"PyTorch  : {torch.__version__}")
    click.echo(f"CUDA     : {'available (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'not available'}")
    click.echo(f"Platform : {platform.platform()}")


@cli.command()
def cite() -> None:
    """Print citation information."""
    print_citation()


@cli.command("init-config")
@click.option("--output", "-o", default="config.yaml", help="Output path for example config.")
def init_config(output: str) -> None:
    """Generate an example configuration file."""
    save_example_config(output)
    click.echo(f"Example configuration written to {output}")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
