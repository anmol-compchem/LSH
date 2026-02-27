"""
Device resolution and reproducibility helpers.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from lsh.logging_utils import get_logger


def resolve_device(requested: str) -> torch.device:
    """
    Resolve a device string to a :class:`torch.device`.

    Parameters
    ----------
    requested : str
        One of ``"cpu"``, ``"cuda"``, or ``"auto"``.

    Returns
    -------
    torch.device
    """
    logger = get_logger()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    get_logger().info("Random seed set to %d (deterministic=%s)", seed, deterministic)
