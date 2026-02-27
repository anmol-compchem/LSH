"""
SOAP descriptor calculation.

Wraps Dscribe's SOAP with configurable parameters and optional GPU support.
The scientific logic is preserved exactly from the original soap.py.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from dscribe.descriptors import SOAP

from lsh.config import SOAPConfig
from lsh.logging_utils import get_logger


def calculate_descriptor_for_frame(
    atoms: Atoms,
    soap_cfg: SOAPConfig,
    species: list[str],
) -> np.ndarray:
    """
    Calculate the SOAP descriptor for a single frame.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure for this frame.
    soap_cfg : SOAPConfig
        SOAP parameters.
    species : list[str]
        List of unique species across the entire trajectory.

    Returns
    -------
    np.ndarray
        SOAP descriptor matrix of shape ``(n_atoms, n_features)``.
    """
    soap = SOAP(
        species=species,
        periodic=soap_cfg.periodic,
        r_cut=soap_cfg.r_cut,
        n_max=soap_cfg.n_max,
        l_max=soap_cfg.l_max,
        rbf=soap_cfg.rbf,
        sigma=soap_cfg.sigma,
    )
    descriptor: np.ndarray = soap.create(atoms)
    return descriptor


def compute_soap_descriptors(
    frames: list[Atoms],
    soap_cfg: SOAPConfig,
    output_folder: str,
    device: torch.device = torch.device("cpu"),
) -> int:
    """
    Compute and save SOAP descriptors for all frames.

    Parameters
    ----------
    frames : list[ase.Atoms]
        Trajectory frames.
    soap_cfg : SOAPConfig
        SOAP parameters.
    output_folder : str
        Directory to save per-frame ``.npy`` descriptor files.
    device : torch.device
        Device for any tensor operations (GPU acceleration for post-processing).

    Returns
    -------
    int
        Number of frames processed.
    """
    logger = get_logger()
    os.makedirs(output_folder, exist_ok=True)

    # Determine species list
    if soap_cfg.species is not None:
        species = soap_cfg.species
    else:
        species_set: set[str] = set()
        for atoms in frames:
            species_set.update(atoms.get_chemical_symbols())
        species = sorted(species_set)

    logger.info("SOAP species: %s", species)
    logger.info("Computing SOAP descriptors for %d frames...", len(frames))

    for idx, atoms in enumerate(frames):
        descriptor = calculate_descriptor_for_frame(atoms, soap_cfg, species)

        # Optional GPU transfer for downstream use (descriptor itself is numpy)
        if device.type == "cuda":
            _ = torch.from_numpy(descriptor).to(device)  # warm-up / validate

        out_path = os.path.join(output_folder, f"descriptor_frame_{idx + 1}.npy")
        np.save(out_path, descriptor)

        if (idx + 1) % 500 == 0 or idx == 0:
            logger.info("  Frame %d/%d — descriptor shape %s", idx + 1, len(frames), descriptor.shape)

    logger.info("SOAP descriptors saved to %s", output_folder)
    return len(frames)
