"""
I/O utilities for reading/writing molecular trajectories.

All operations go through ASE's Atoms objects, making the pipeline
format-agnostic.  Input can be any ASE-readable format (XYZ, extXYZ,
LAMMPS, CIF, VASP, PDB, GRO, …).  Output format is configurable via
``io.output_format`` in the YAML config.

NPT support
-----------
Under NPT, each frame has a different cell.  The ``cell`` config override
is applied ONLY to frames that have no cell (all-zero or missing).  Frames
that already carry cell information from the trajectory file are left
untouched.
"""

from __future__ import annotations

import os
import random as _random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from lsh.logging_utils import get_logger


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
_FORMAT_FROM_SUFFIX: dict[str, str] = {
    ".xyz": "extxyz",
    ".extxyz": "extxyz",
    ".gro": "gro",
    ".lammpstrj": "lammps-dump-text",
    ".dump": "lammps-dump-text",
    # NOTE: .lmp is intentionally excluded — it's ambiguous between
    # lammps-data (single structure) and lammps-dump-text (trajectory).
    # We sniff the file content instead. See _guess_format().
    ".cif": "cif",
    ".pdb": "proteindatabank",
    ".vasp": "vasp",
    ".poscar": "vasp",
    ".contcar": "vasp",
}

_EXTENSION_FROM_FORMAT: dict[str, str] = {
    "extxyz": ".xyz",
    "xyz": ".xyz",
    "gro": ".gro",
    "lammps-data": ".lmp",
    "lammps-dump-text": ".lammpstrj",
    "cif": ".cif",
    "vasp": ".vasp",
    "proteindatabank": ".pdb",
}


def _guess_format(file_path: str) -> Optional[str]:
    """Guess ASE format from file extension, with content sniffing for ambiguous cases."""
    suffix = Path(file_path).suffix.lower()

    # Direct mapping for unambiguous extensions
    fmt = _FORMAT_FROM_SUFFIX.get(suffix)
    if fmt is not None:
        return fmt

    # .lmp is ambiguous — sniff first non-blank line
    if suffix == ".lmp":
        return _sniff_lammps_format(file_path)

    return None


def _sniff_lammps_format(file_path: str) -> str:
    """Read first few lines to distinguish lammps-data from lammps-dump-text."""
    try:
        with open(file_path, "r") as fh:
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                if "ITEM:" in line:
                    return "lammps-dump-text"
        return "lammps-data"
    except Exception:
        return "lammps-data"


def _format_to_extension(fmt: str) -> str:
    """Map ASE format names to file extensions."""
    return _EXTENSION_FROM_FORMAT.get(fmt, f".{fmt}")


def _has_cell(atoms: Atoms) -> bool:
    """Return True if atoms has a non-trivial cell (any non-zero component)."""
    return atoms.cell.any()


# ---------------------------------------------------------------------------
# Universal trajectory reader
# ---------------------------------------------------------------------------
def read_trajectory(
    file_path: str,
    fmt: str = "auto",
    cell: Optional[list[float]] = None,
    pbc: Optional[bool] = None,
) -> list[Atoms]:
    """
    Read a molecular trajectory using ASE.

    For NPT trajectories, per-frame cell information from the file is
    preserved.  The ``cell`` override is applied **only** to frames that
    have no cell (all-zero), not to frames that already carry cell data.

    Parameters
    ----------
    file_path : str
        Path to trajectory file.
    fmt : str
        File format hint for ASE (``"auto"`` to auto-detect).
    cell : list[float], optional
        Fallback cell for frames missing cell info.
    pbc : bool, optional
        Override periodic boundary conditions flag.

    Returns
    -------
    list[ase.Atoms]
    """
    logger = get_logger()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    # Resolve format
    if fmt == "auto":
        guessed = _guess_format(file_path)
        kwargs: dict = {"index": ":"}
        if guessed is not None:
            kwargs["format"] = guessed
            logger.info("Reading trajectory: %s (detected format=%s)", file_path, guessed)
        else:
            logger.info("Reading trajectory: %s (ASE auto-detect)", file_path)
    else:
        kwargs = {"index": ":", "format": fmt}
        logger.info("Reading trajectory: %s (format=%s)", file_path, fmt)

    frames: list[Atoms] = ase_read(str(path), **kwargs)

    # ── Apply cell override only to frames missing cell info ──────────
    if cell is not None:
        n_applied = 0
        n_preserved = 0
        for atoms in frames:
            if _has_cell(atoms):
                n_preserved += 1
            else:
                atoms.set_cell(cell)
                n_applied += 1
        if n_preserved > 0 and n_applied > 0:
            logger.warning(
                "Cell override: applied to %d frames without cell, "
                "preserved existing cell on %d frames",
                n_applied, n_preserved,
            )
        elif n_preserved > 0:
            logger.info(
                "Cell override ignored — all %d frames already have cell info "
                "(NPT trajectory detected)",
                n_preserved,
            )
        else:
            logger.info("Cell override applied to all %d frames", n_applied)

    # ── Apply PBC override ────────────────────────────────────────────
    if pbc is not None:
        for atoms in frames:
            atoms.set_pbc(pbc)

    logger.info("Loaded %d frames from %s", len(frames), path.name)
    return frames


def validate_frames_for_soap(
    frames: list[Atoms],
    periodic: bool,
) -> None:
    """
    Check that frames are suitable for SOAP computation.

    Raises warnings or errors for common pitfalls:
    - periodic=True but frames have no cell
    - varying atom counts across frames

    Parameters
    ----------
    frames : list[ase.Atoms]
        Trajectory frames.
    periodic : bool
        Whether periodic SOAP is requested.
    """
    logger = get_logger()

    if not frames:
        raise ValueError("Trajectory is empty — no frames loaded")

    # ── Check cell availability for periodic SOAP ─────────────────────
    if periodic:
        n_no_cell = sum(1 for f in frames if not _has_cell(f))
        if n_no_cell == len(frames):
            raise ValueError(
                "soap.periodic=true but no frame has cell information. "
                "Either set io.cell in config, or use a format that "
                "carries cell data (extxyz, gro, lammps-dump-text, cif, …)"
            )
        if n_no_cell > 0:
            logger.warning(
                "%d / %d frames have no cell — periodic SOAP will fail "
                "on those frames. Consider setting io.cell as fallback.",
                n_no_cell, len(frames),
            )

    # ── Check for varying atom counts ─────────────────────────────────
    atom_counts = {len(f) for f in frames}
    if len(atom_counts) > 1:
        logger.warning(
            "Trajectory has varying atom counts: %s. "
            "Flattened SOAP descriptors will have different lengths — "
            "the flatten-and-stack step will pad shorter descriptors with zeros.",
            sorted(atom_counts),
        )

    # ── Log cell variation for NPT awareness ──────────────────────────
    cells_with_info = [f for f in frames if _has_cell(f)]
    if len(cells_with_info) >= 2:
        vols = [f.cell.volume for f in cells_with_info]
        vol_min, vol_max = min(vols), max(vols)
        if vol_max > 0 and (vol_max - vol_min) / vol_max > 0.001:
            logger.info(
                "NPT trajectory detected: cell volume varies from "
                "%.1f to %.1f Å³ (%.2f%% variation)",
                vol_min, vol_max, 100 * (vol_max - vol_min) / vol_max,
            )


# ---------------------------------------------------------------------------
# Frame selection from bins  (format-agnostic, index-based)
# ---------------------------------------------------------------------------
def select_representative_frames(
    bin_to_frames: dict[int, list[int]],
    method: str = "medoid",
    descriptor_folder: Optional[str] = None,
    random_seed: int = 42,
) -> list[int]:
    """
    Select one representative frame per LSH bin.

    Parameters
    ----------
    bin_to_frames : dict[int, list[int]]
        Mapping ``{bin_id: [frame_indices]}``.
    method : str
        Selection strategy:

        - ``"first"``  — first frame in each bin (original behaviour).
        - ``"random"`` — uniformly random frame per bin.
        - ``"medoid"`` — frame whose descriptor is closest to the bin
          centroid (most representative).  Falls back to ``"first"`` if
          descriptors are unavailable.
    descriptor_folder : str, optional
        Path to per-frame ``.npy`` descriptor files (required for medoid).
    random_seed : int
        Seed for ``"random"`` method.

    Returns
    -------
    list[int]
        Selected frame indices (one per bin, sorted).
    """
    logger = get_logger()

    if method == "first":
        selected = [frames[0] for frames in bin_to_frames.values()]

    elif method == "random":
        rng = _random.Random(random_seed)
        selected = [rng.choice(frames) for frames in bin_to_frames.values()]

    elif method == "medoid":
        if descriptor_folder is None or not os.path.isdir(descriptor_folder):
            logger.warning(
                "Descriptor folder not available for medoid selection; "
                "falling back to 'first'"
            )
            selected = [frames[0] for frames in bin_to_frames.values()]
        else:
            selected = _select_medoids(bin_to_frames, descriptor_folder, logger)
    else:
        raise ValueError(f"Unknown selection method: {method!r}")

    selected.sort()
    logger.info(
        "Selected %d representative frames (method=%s)", len(selected), method
    )
    return selected


def _select_medoids(
    bin_to_frames: dict[int, list[int]],
    descriptor_folder: str,
    logger,
) -> list[int]:
    """Pick the frame closest to the bin centroid in descriptor space."""
    selected: list[int] = []

    for bin_id, frame_indices in bin_to_frames.items():
        if len(frame_indices) == 1:
            selected.append(frame_indices[0])
            continue

        # Load descriptors for frames in this bin
        descriptors = []
        valid_indices = []
        for fidx in frame_indices:
            path = os.path.join(
                descriptor_folder, f"descriptor_frame_{fidx + 1}.npy"
            )
            if os.path.exists(path):
                descriptors.append(np.load(path).flatten())
                valid_indices.append(fidx)

        if not descriptors:
            selected.append(frame_indices[0])
            continue

        # Zero-pad if descriptors have different lengths (variable atom count)
        max_len = max(len(d) for d in descriptors)
        padded = []
        for d in descriptors:
            if len(d) < max_len:
                d = np.pad(d, (0, max_len - len(d)), mode="constant")
            padded.append(d)

        mat = np.stack(padded, axis=0)              # (n_in_bin, feat_dim)
        centroid = mat.mean(axis=0, keepdims=True)  # (1, feat_dim)

        # Euclidean distance to centroid
        dists = np.linalg.norm(mat - centroid, axis=1)
        best_local = int(np.argmin(dists))
        selected.append(valid_indices[best_local])

    return selected


# ---------------------------------------------------------------------------
# Frame extraction  (format-agnostic via ASE)
# ---------------------------------------------------------------------------
def extract_frames(
    frames: list[Atoms],
    frame_indices: Sequence[int],
    output_file: str,
    output_format: str = "extxyz",
) -> int:
    """
    Write selected frames to a trajectory file in any ASE-supported format.

    Parameters
    ----------
    frames : list[ase.Atoms]
        Full trajectory (already loaded in memory from Step 1).
    frame_indices : Sequence[int]
        Indices of frames to extract.
    output_file : str
        Destination file path.
    output_format : str
        ASE output format (``"extxyz"``, ``"xyz"``, ``"gro"``,
        ``"lammps-data"``, ``"cif"``, ``"vasp"``, etc.).

    Returns
    -------
    int
        Number of frames written.
    """
    logger = get_logger()
    selected = [frames[i] for i in frame_indices if i < len(frames)]
    ase_write(output_file, selected, format=output_format)
    logger.info(
        "Extracted %d frames → %s (format=%s)",
        len(selected), output_file, output_format,
    )
    return len(selected)


# ---------------------------------------------------------------------------
# Trajectory splitting  (format-agnostic via ASE)
# ---------------------------------------------------------------------------
def split_trajectory(
    input_file: str,
    frames_per_file: int,
    output_dir: str,
    output_format: str = "extxyz",
    input_format: Optional[str] = None,
) -> int:
    """
    Split a trajectory into smaller files.

    Parameters
    ----------
    input_file : str
        Input trajectory file.
    frames_per_file : int
        Maximum frames per output file.
    output_dir : str
        Directory for output part files.
    output_format : str
        ASE output format for the split files.
    input_format : str, optional
        Explicit format hint for reading (None = auto-detect).

    Returns
    -------
    int
        Number of files written.
    """
    logger = get_logger()
    os.makedirs(output_dir, exist_ok=True)

    ext = _format_to_extension(output_format)

    kwargs: dict = {"index": ":"}
    if input_format:
        kwargs["format"] = input_format

    frames: list[Atoms] = ase_read(input_file, **kwargs)
    n_frames = len(frames)
    file_index = 0

    for start in range(0, n_frames, frames_per_file):
        chunk = frames[start : start + frames_per_file]
        file_index += 1
        out_path = os.path.join(output_dir, f"part_{file_index}{ext}")
        ase_write(out_path, chunk, format=output_format)
        logger.info("Written %s with %d frames", out_path, len(chunk))

    logger.info("Split complete: %d files in %s", file_index, output_dir)
    return file_index


# ---------------------------------------------------------------------------
# Write frame indices to file
# ---------------------------------------------------------------------------
def write_frame_dat(frame_numbers: Sequence[int], output_file: str) -> None:
    """Write frame numbers to a ``frame.dat`` file."""
    with open(output_file, "w") as fh:
        for n in frame_numbers:
            fh.write(f"{n}\n")
    get_logger().info("Wrote %d frame numbers to %s", len(frame_numbers), output_file)
