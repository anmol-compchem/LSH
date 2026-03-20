"""
Structured logging for LSH-DP pipeline.

Creates both console (rich) and file handlers with detailed run metadata.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from lsh import __version__

_LOGGER_NAME = "lshdp"


def setup_logging(
    output_dir: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return the package logger.

    Parameters
    ----------
    output_dir : str
        Directory for the log file.
    log_file : str, optional
        Explicit log filename. Auto-generated if *None*.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler ---------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"lshdp_run_{timestamp}.log"
    log_path = Path(output_dir) / log_file

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(level)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.info("LSH-DP v%s — log initialised", __version__)
    return logger


def get_logger() -> logging.Logger:
    """Return the package logger (must call :func:`setup_logging` first)."""
    return logging.getLogger(_LOGGER_NAME)


def log_hardware_info(logger: logging.Logger, device: str) -> None:
    """Write hardware / environment metadata to the log."""
    logger.info("Platform        : %s", platform.platform())
    logger.info("Python          : %s", sys.version.split()[0])
    logger.info("PyTorch         : %s", torch.__version__)
    logger.info("CUDA available  : %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device     : %s", torch.cuda.get_device_name(0))
        logger.info("CUDA version    : %s", torch.version.cuda)
    logger.info("Selected device : %s", device)


def log_config_summary(logger: logging.Logger, cfg) -> None:
    """Write a compact configuration summary."""
    logger.info("--- Configuration Summary ---")
    logger.info("Input file      : %s", cfg.io.input_file)
    logger.info("Output dir      : %s", cfg.io.output_dir)
    logger.info("Input format    : %s", cfg.io.format)
    logger.info("Output format   : %s", cfg.io.output_format)
    logger.info("SOAP r_cut      : %s", cfg.soap.r_cut)
    logger.info("SOAP n_max      : %s", cfg.soap.n_max)
    logger.info("SOAP l_max      : %s", cfg.soap.l_max)
    logger.info("SOAP sigma      : %s", cfg.soap.sigma)
    logger.info("SOAP periodic   : %s", cfg.soap.periodic)
    logger.info("SOAP n_jobs     : %s", cfg.soap.n_jobs)
    logger.info("PCA components  : %s", cfg.hashing.n_components)
    logger.info("Hash functions  : %s", cfg.hashing.n_hash)
    logger.info("Bin width       : %s", cfg.hashing.bin_width)
    logger.info("Random seed     : %s", cfg.hashing.random_seed)
    logger.info("Selection       : %s", cfg.selection.method)
    logger.info("Split frames    : %s", cfg.split.frames_per_file)
    logger.info("Device          : %s", cfg.device)
    logger.info("Deterministic   : %s", cfg.deterministic)
    logger.info("Steps           : %d → %d", cfg.start_step, cfg.end_step)
    logger.info("-----------------------------")
