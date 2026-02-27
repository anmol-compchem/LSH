"""
I/O utilities for reading/writing molecular trajectories.

Supports any ASE-readable format and provides XYZ-specific utilities
for metadata updates, frame extraction, and splitting.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read as ase_read

from lsh.logging_utils import get_logger


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

    Parameters
    ----------
    file_path : str
        Path to trajectory file.
    fmt : str
        File format hint for ASE (``"auto"`` to auto-detect).
    cell : list[float], optional
        Override unit cell, e.g. ``[15.82, 15.82, 30.76]``.
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

    logger.info("Reading trajectory: %s (format=%s)", file_path, fmt)

    kwargs: dict = {"index": ":"}
    if fmt != "auto":
        kwargs["format"] = fmt

    frames: list[Atoms] = ase_read(str(path), **kwargs)

    # Apply overrides
    if cell is not None:
        for atoms in frames:
            atoms.set_cell(cell)
    if pbc is not None:
        for atoms in frames:
            atoms.set_pbc(pbc)

    logger.info("Loaded %d frames from %s", len(frames), path.name)
    return frames


def frames_to_dataframes(frames: list[Atoms]) -> list[pd.DataFrame]:
    """Convert ASE Atoms objects to DataFrames with Element/x/y/z columns."""
    dfs: list[pd.DataFrame] = []
    for atoms in frames:
        df = pd.DataFrame({
            "Element": atoms.get_chemical_symbols(),
            "x": atoms.positions[:, 0],
            "y": atoms.positions[:, 1],
            "z": atoms.positions[:, 2],
        })
        dfs.append(df)
    return dfs


# ---------------------------------------------------------------------------
# XYZ metadata update  (replaces framevalue.py)
# ---------------------------------------------------------------------------
def update_xyz_metadata(input_file: str, output_file: str) -> int:
    """
    Rewrite an XYZ file with standardised ``i = <N>, time = <T>`` metadata.

    Parameters
    ----------
    input_file : str
        Source XYZ file.
    output_file : str
        Destination XYZ file.

    Returns
    -------
    int
        Number of frames processed.
    """
    logger = get_logger()
    frame_index = 0

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        while True:
            natoms_line = fin.readline()
            if not natoms_line:
                break
            natoms_line = natoms_line.strip()
            if natoms_line == "":
                continue
            natoms = int(natoms_line)

            fout.write(natoms_line + "\n")

            # Skip existing metadata
            _ = fin.readline()

            new_meta = f"i = {frame_index:8d}, time = {frame_index * 0.5:8.3f}"
            fout.write(new_meta + "\n")

            for _ in range(natoms):
                coord_line = fin.readline()
                if not coord_line:
                    raise EOFError("Unexpected end of file while reading coordinates")
                fout.write(coord_line)

            frame_index += 1

    logger.info("Updated metadata for %d frames → %s", frame_index, output_file)
    return frame_index


# ---------------------------------------------------------------------------
# Frame extraction  (replaces extract_frame.py + frame.py)
# ---------------------------------------------------------------------------
def extract_frame_numbers_from_bins(bin_file: str) -> list[int]:
    """
    Parse an output-bins file and return one representative frame per bin.

    Parameters
    ----------
    bin_file : str
        Path to ``output_bins_<width>.txt``.

    Returns
    -------
    list[int]
        Extracted frame numbers (one per bin).
    """
    logger = get_logger()
    frame_numbers: list[int] = []

    with open(bin_file, "r") as fh:
        for line in fh:
            if "[" in line and "]" in line:
                fields = re.split(r"[\[\],]", line.strip())
                if len(fields) > 2:
                    frame_numbers.append(int(fields[1].strip()))

    logger.info("Extracted %d representative frame numbers from %s", len(frame_numbers), bin_file)
    return frame_numbers


def write_frame_dat(frame_numbers: Sequence[int], output_file: str) -> None:
    """Write frame numbers to a ``frame.dat`` file."""
    with open(output_file, "w") as fh:
        for n in frame_numbers:
            fh.write(f"{n}\n")
    get_logger().info("Wrote %d frame numbers to %s", len(frame_numbers), output_file)


def extract_frames_from_xyz(
    input_file: str,
    frame_numbers: set[int],
    output_file: str,
) -> int:
    """
    Extract specific frames from a metadata-tagged XYZ file.

    Parameters
    ----------
    input_file : str
        XYZ file with ``i = <N>`` metadata (output of :func:`update_xyz_metadata`).
    frame_numbers : set[int]
        Frame indices to extract.
    output_file : str
        Destination XYZ file.

    Returns
    -------
    int
        Number of frames extracted.
    """
    logger = get_logger()
    extracted = 0

    with open(input_file, "r") as fin:
        lines = fin.readlines()

    with open(output_file, "w") as fout:
        i = 0
        while i < len(lines):
            try:
                num_atoms = int(lines[i].strip())
            except (ValueError, IndexError):
                i += 1
                continue

            if i + 1 >= len(lines):
                break

            metadata_line = lines[i + 1].strip()
            match = re.search(r"i =\s*(\d+)", metadata_line)

            if match:
                current = int(match.group(1))
                if current in frame_numbers:
                    fout.write(f"{num_atoms}\n")
                    fout.write(metadata_line + "\n")
                    for j in range(i + 2, min(i + 2 + num_atoms, len(lines))):
                        fout.write(lines[j])
                    extracted += 1

            i += 2 + num_atoms

    logger.info("Extracted %d frames → %s", extracted, output_file)
    return extracted


# ---------------------------------------------------------------------------
# XYZ splitting  (replaces split.py)
# ---------------------------------------------------------------------------
def split_xyz(input_file: str, frames_per_file: int, output_dir: str) -> int:
    """
    Split an XYZ trajectory into smaller files.

    Parameters
    ----------
    input_file : str
        Input XYZ file.
    frames_per_file : int
        Maximum frames per output file.
    output_dir : str
        Directory for output part files.

    Returns
    -------
    int
        Number of files written.
    """
    logger = get_logger()
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as fh:
        lines = fh.readlines()

    idx = 0
    frame_count = 0
    file_index = 1
    current: list[str] = []

    while idx < len(lines):
        try:
            n_atoms = int(lines[idx].strip())
        except ValueError:
            break
        frame_size = n_atoms + 2
        current.extend(lines[idx: idx + frame_size])
        frame_count += 1
        idx += frame_size

        if frame_count == frames_per_file:
            out_path = os.path.join(output_dir, f"part_{file_index}.xyz")
            with open(out_path, "w") as out:
                out.writelines(current)
            logger.info("Written %s with %d frames", out_path, frame_count)
            file_index += 1
            frame_count = 0
            current = []

    if current:
        out_path = os.path.join(output_dir, f"part_{file_index}.xyz")
        with open(out_path, "w") as out:
            out.writelines(current)
        logger.info("Written %s with %d frames", out_path, frame_count)
        file_index += 1

    total_files = file_index - 1
    logger.info("Split complete: %d files in %s", total_files, output_dir)
    return total_files
