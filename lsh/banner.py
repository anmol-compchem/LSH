"""
Startup banner for LSH-DP.
"""

from __future__ import annotations

from lsh import __version__, __author__, __citation__


def print_banner(cfg=None) -> None:
    """Print the startup banner to stdout."""
    width = 64
    sep = "=" * width
    thin = "-" * width

    lines = [
        sep,
        " Locality-Sensitive Hashing Dataset Reduction (LSH-DP)".center(width),
        f" v{__version__}".center(width),
        "",
        f"  Authors : {__author__}",
        "  Journal : J. Chem. Theory Comput. 2025, 21, 12, 6113–6120",
        sep,
    ]

    if cfg is not None:
        lines += [
            f"  Input file      : {cfg.io.input_file}",
            f"  Output dir      : {cfg.io.output_dir}",
            f"  Device          : {cfg.device}",
            f"  SOAP r_cut      : {cfg.soap.r_cut}",
            f"  SOAP n_max/l_max: {cfg.soap.n_max} / {cfg.soap.l_max}",
            f"  PCA components  : {cfg.hashing.n_components}",
            f"  Bin width       : {cfg.hashing.bin_width}",
            f"  Steps           : {cfg.start_step} → {cfg.end_step}",
            thin,
        ]

    print("\n".join(lines))


def print_citation() -> None:
    """Print the citation block."""
    print()
    print("If you use this software, please cite:")
    print()
    print(f"  {__citation__}")
    print()
    print(
        "  This implementation follows the algorithm described in:\n"
        '  "Locality-Sensitive Hashing-Based Data Set Reduction\n'
        '   for Deep Potential Training"'
    )
    print()
