"""
SOAP descriptor calculation.

Wraps DScribe's SOAP with configurable parameters and parallel batch support.
The scientific logic is preserved exactly from the original soap.py.

Performance note
----------------
DScribe's ``SOAP.create()`` accepts a **list** of ``Atoms`` objects and
parallelises the computation across them via joblib (``n_jobs`` parameter).
This is the single highest-impact optimisation available — it avoids
re-constructing the SOAP object per frame and uses all available CPU cores.

GPU note
--------
DScribe computes SOAP descriptors on the CPU; there is no GPU kernel.
The ``device`` argument is forwarded to the hashing / LSH stages only.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

from lsh.config import SOAPConfig
from lsh.logging_utils import get_logger


def _build_soap(soap_cfg: SOAPConfig, species: list[str]) -> SOAP:
    """Construct a reusable SOAP descriptor object."""
    return SOAP(
        species=species,
        periodic=soap_cfg.periodic,
        r_cut=soap_cfg.r_cut,
        n_max=soap_cfg.n_max,
        l_max=soap_cfg.l_max,
        rbf=soap_cfg.rbf,
        sigma=soap_cfg.sigma,
    )


def calculate_descriptor_for_frame(
    atoms: Atoms,
    soap_cfg: SOAPConfig,
    species: list[str],
) -> np.ndarray:
    """
    Calculate the SOAP descriptor for a single frame (convenience wrapper).

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
    soap = _build_soap(soap_cfg, species)
    descriptor: np.ndarray = soap.create(atoms)
    return descriptor


def compute_soap_descriptors(
    frames: list[Atoms],
    soap_cfg: SOAPConfig,
    output_folder: str,
    **_kwargs,
) -> int:
    """
    Compute and save SOAP descriptors for all frames using parallel batch mode.

    This replaces the original serial loop with DScribe's native batch
    interface: ``soap.create(list_of_atoms, n_jobs=N)``.  On a machine with
    *C* cores and *F* frames, wall-clock time drops from O(F) to roughly
    O(F / C).

    Parameters
    ----------
    frames : list[ase.Atoms]
        Trajectory frames.
    soap_cfg : SOAPConfig
        SOAP parameters (includes ``n_jobs``).
    output_folder : str
        Directory to save per-frame ``.npy`` descriptor files.

    Returns
    -------
    int
        Number of frames processed.
    """
    logger = get_logger()
    os.makedirs(output_folder, exist_ok=True)

    # ── Resolve species ──────────────────────────────────────────────
    if soap_cfg.species is not None:
        species = soap_cfg.species
    else:
        species_set: set[str] = set()
        for atoms in frames:
            species_set.update(atoms.get_chemical_symbols())
        species = sorted(species_set)

    logger.info("SOAP species: %s", species)

    # ── Build descriptor ONCE ────────────────────────────────────────
    soap = _build_soap(soap_cfg, species)

    n_jobs = soap_cfg.n_jobs
    n_frames = len(frames)
    logger.info(
        "Computing SOAP descriptors for %d frames (n_jobs=%s) ...",
        n_frames,
        n_jobs,
    )

    # ── Batch computation ────────────────────────────────────────────
    # DScribe's create() returns a list of np.ndarray when given a list
    # of Atoms. joblib parallelism is controlled by n_jobs.
    descriptors = soap.create(frames, n_jobs=n_jobs)

    # ── Save per-frame .npy files (I/O is fast, keep sequential) ─────
    for idx, descriptor in enumerate(descriptors):
        out_path = os.path.join(output_folder, f"descriptor_frame_{idx + 1}.npy")
        np.save(out_path, descriptor)

        if (idx + 1) % 500 == 0 or idx == 0:
            logger.info(
                "  Saved frame %d/%d — descriptor shape %s",
                idx + 1,
                n_frames,
                descriptor.shape,
            )

    logger.info("SOAP descriptors saved to %s", output_folder)
    return n_frames
