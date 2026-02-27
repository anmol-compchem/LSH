"""
Configuration management for LSH-DP pipeline.

Handles loading, validation, and merging of YAML config files with CLI overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SOAPConfig:
    """SOAP descriptor parameters."""
    r_cut: float = 6.0
    n_max: int = 4
    l_max: int = 4
    sigma: float = 1.0
    rbf: str = "gto"
    periodic: bool = True
    species: Optional[list[str]] = None  # auto-detected if None


@dataclass
class HashingConfig:
    """Locality-sensitive hashing parameters."""
    n_components: int = 100
    n_hash: int = 100
    bin_width: float = 0.004
    random_seed: int = 42


@dataclass
class IOConfig:
    """Input / output paths and format settings."""
    input_file: str = "simulation.xyz"
    output_dir: str = "results"
    format: str = "auto"  # "auto" lets ASE detect
    cell: Optional[list[float]] = None  # override cell, e.g. [15.82, 15.82, 30.76]
    pbc: Optional[bool] = None  # override PBC flag


@dataclass
class SplitConfig:
    """Frame splitting settings."""
    frames_per_file: int = 120


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    soap: SOAPConfig = field(default_factory=SOAPConfig)
    hashing: HashingConfig = field(default_factory=HashingConfig)
    io: IOConfig = field(default_factory=IOConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    device: str = "auto"  # "cpu", "cuda", "auto"
    deterministic: bool = True
    start_step: int = 1
    end_step: int = 7
    log_file: Optional[str] = None  # auto-generated if None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _merge_dict(target: dict, source: dict) -> dict:
    """Recursively merge *source* into *target*."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(config_path: Optional[str] = None, overrides: Optional[dict[str, Any]] = None) -> PipelineConfig:
    """
    Load configuration from YAML file and apply CLI overrides.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file.
    overrides : dict, optional
        Key-value overrides (e.g. from CLI flags).

    Returns
    -------
    PipelineConfig
    """
    raw: dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

    if overrides:
        _merge_dict(raw, overrides)

    soap_cfg = _dataclass_from_dict(SOAPConfig, raw.get("soap", {}))
    hash_cfg = _dataclass_from_dict(HashingConfig, raw.get("hashing", {}))
    io_cfg = _dataclass_from_dict(IOConfig, raw.get("io", {}))
    split_cfg = _dataclass_from_dict(SplitConfig, raw.get("split", {}))

    cfg = PipelineConfig(
        soap=soap_cfg,
        hashing=hash_cfg,
        io=io_cfg,
        split=split_cfg,
        device=raw.get("device", "auto"),
        deterministic=raw.get("deterministic", True),
        start_step=raw.get("start_step", 1),
        end_step=raw.get("end_step", 7),
        log_file=raw.get("log_file", None),
    )
    return cfg


def validate_config(cfg: PipelineConfig) -> list[str]:
    """
    Validate a PipelineConfig and return a list of warnings/errors.

    Returns
    -------
    list[str]
        Empty list means the config is valid.
    """
    issues: list[str] = []

    if cfg.soap.r_cut <= 0:
        issues.append("soap.r_cut must be positive")
    if cfg.soap.n_max < 1:
        issues.append("soap.n_max must be >= 1")
    if cfg.soap.l_max < 0:
        issues.append("soap.l_max must be >= 0")
    if cfg.soap.sigma <= 0:
        issues.append("soap.sigma must be positive")
    if cfg.hashing.n_components < 1:
        issues.append("hashing.n_components must be >= 1")
    if cfg.hashing.n_hash < 1:
        issues.append("hashing.n_hash must be >= 1")
    if cfg.hashing.bin_width <= 0:
        issues.append("hashing.bin_width must be positive")
    if cfg.device not in ("cpu", "cuda", "auto"):
        issues.append(f"device must be 'cpu', 'cuda', or 'auto'; got '{cfg.device}'")
    if not (1 <= cfg.start_step <= 7):
        issues.append("start_step must be between 1 and 7")
    if not (1 <= cfg.end_step <= 7):
        issues.append("end_step must be between 1 and 7")
    if cfg.start_step > cfg.end_step:
        issues.append("start_step cannot be greater than end_step")
    if cfg.split.frames_per_file < 1:
        issues.append("split.frames_per_file must be >= 1")

    return issues


def save_example_config(path: str) -> None:
    """Write an example configuration YAML to *path*."""
    example = {
        "soap": {
            "r_cut": 6.0,
            "n_max": 4,
            "l_max": 4,
            "sigma": 1.0,
            "rbf": "gto",
            "periodic": True,
        },
        "hashing": {
            "n_components": 100,
            "n_hash": 100,
            "bin_width": 0.004,
            "random_seed": 42,
        },
        "io": {
            "input_file": "simulation.xyz",
            "output_dir": "results",
            "format": "auto",
        },
        "split": {
            "frames_per_file": 120,
        },
        "device": "auto",
        "deterministic": True,
    }
    with open(path, "w") as fh:
        yaml.dump(example, fh, default_flow_style=False, sort_keys=False)
