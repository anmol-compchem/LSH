"""
Pipeline orchestrator — runs all steps in sequence.

Steps 1–3 are unchanged (SOAP → LSH → bins).
Steps 4–6 are format-agnostic: frame selection and
extraction go through ASE Atoms objects, so the pipeline works with
any input format (LAMMPS, CIF, VASP, XYZ, GRO, …) without
format-specific text parsing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from lsh.config import PipelineConfig
from lsh.logging_utils import get_logger
from lsh.utils import resolve_device, set_seed
from lsh.descriptors import compute_soap_descriptors
from lsh.hashing import (
    flatten_and_stack_descriptors,
    reduce_dimensionality,
    process_with_hashing,
    organise_bins,
)
from lsh.io import (
    read_trajectory,
    validate_frames_for_soap,
    select_representative_frames,
    extract_frames,
    write_frame_dat,
    split_trajectory,
)


def _step_active(cfg: PipelineConfig, step: int) -> bool:
    return cfg.start_step <= step <= cfg.end_step


def run_pipeline(cfg: PipelineConfig) -> None:
    """
    Execute the full LSH-DP pipeline.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully resolved configuration.
    """
    logger = get_logger()
    device = resolve_device(cfg.device)
    set_seed(cfg.hashing.random_seed, deterministic=cfg.deterministic)

    out = cfg.io.output_dir
    os.makedirs(out, exist_ok=True)

    descriptor_folder = os.path.join(out, "descriptors")
    hash_folder = os.path.join(out, "hash_buckets")
    parts_folder = os.path.join(out, "parts")

    bw = cfg.hashing.bin_width
    n_comp = cfg.hashing.n_components
    out_fmt = cfg.io.output_format

    timings: dict[str, float] = {}

    # We may need the full trajectory in steps 1 and 5. Load once, reuse.
    frames: list | None = None
    bin_to_frames: dict[int, list[int]] | None = None

    def _load_frames():
        nonlocal frames
        if frames is None:
            frames = read_trajectory(
                cfg.io.input_file,
                fmt=cfg.io.format,
                cell=cfg.io.cell,
                pbc=cfg.io.pbc,
            )
            logger.info("Frames loaded: %d", len(frames))
        return frames

    # ------------------------------------------------------------------
    # Step 1: SOAP descriptors
    # ------------------------------------------------------------------
    if _step_active(cfg, 1):
        logger.info("=== Step 1: Computing SOAP descriptors ===")
        t0 = time.time()

        _load_frames()
        validate_frames_for_soap(frames, periodic=cfg.soap.periodic)
        compute_soap_descriptors(frames, cfg.soap, descriptor_folder)

        timings["step1_soap"] = time.time() - t0
        logger.info("Step 1 completed in %.1f s", timings["step1_soap"])

    # ------------------------------------------------------------------
    # Step 2: LSH (flatten → PCA → hash → buckets)
    # ------------------------------------------------------------------
    if _step_active(cfg, 2):
        logger.info("=== Step 2: Locality-Sensitive Hashing ===")
        t0 = time.time()

        combined_file = os.path.join(hash_folder, "combined_tensor.npy")
        if os.path.exists(combined_file):
            logger.info("Loading existing combined tensor")
            combined = torch.from_numpy(np.load(combined_file)).to(device)
        else:
            combined = flatten_and_stack_descriptors(descriptor_folder, hash_folder, device)

        reduced_file = os.path.join(hash_folder, f"reduced_tensor_{n_comp}.npy")
        if os.path.exists(reduced_file):
            logger.info("Loading existing reduced tensor")
            reduced = torch.from_numpy(np.load(reduced_file)).float().to(device)
        else:
            reduced = reduce_dimensionality(combined, n_comp, hash_folder, device)

        buckets_file = os.path.join(hash_folder, f"hash_buckets_flattened_{bw}.txt")
        if not os.path.exists(buckets_file):
            process_with_hashing(reduced, cfg.hashing, hash_folder, device)
        else:
            logger.info("Hash buckets already exist, skipping")

        timings["step2_hashing"] = time.time() - t0
        logger.info("Step 2 completed in %.1f s", timings["step2_hashing"])

    # ------------------------------------------------------------------
    # Step 3: Organise bins
    # ------------------------------------------------------------------
    if _step_active(cfg, 3):
        logger.info("=== Step 3: Organising bins ===")
        t0 = time.time()

        buckets_file = os.path.join(hash_folder, f"hash_buckets_flattened_{bw}.txt")
        bins_file = os.path.join(hash_folder, f"output_bins_{bw}.txt")
        bin_to_frames = organise_bins(buckets_file, bins_file)

        timings["step3_distil"] = time.time() - t0
        logger.info("Step 3 completed in %.1f s", timings["step3_distil"])

    # ------------------------------------------------------------------
    # Step 4: Select representative frames from bins
    # ------------------------------------------------------------------
    if _step_active(cfg, 4):
        logger.info("=== Step 4: Selecting representative frames ===")
        t0 = time.time()

        # Reload bin mapping if step 3 was skipped
        if bin_to_frames is None:
            bins_file = os.path.join(hash_folder, f"output_bins_{bw}.txt")
            bin_to_frames = _parse_bins_file(bins_file)

        frame_indices = select_representative_frames(
            bin_to_frames,
            method=cfg.selection.method,
            descriptor_folder=descriptor_folder,
            random_seed=cfg.selection.random_seed,
        )

        frame_dat = os.path.join(out, "frame.dat")
        write_frame_dat(frame_indices, frame_dat)

        timings["step4_select"] = time.time() - t0
        logger.info("Step 4 completed in %.1f s", timings["step4_select"])

    # ------------------------------------------------------------------
    # Step 5: Extract selected frames (format-agnostic)
    # ------------------------------------------------------------------
    if _step_active(cfg, 5):
        logger.info("=== Step 5: Extracting selected frames ===")
        t0 = time.time()

        _load_frames()

        frame_dat = os.path.join(out, "frame.dat")
        if not os.path.exists(frame_dat):
            raise FileNotFoundError(f"frame.dat not found at {frame_dat}; run step 4 first")

        with open(frame_dat, "r") as fh:
            frame_indices = sorted(int(line.strip()) for line in fh if line.strip())

        trj_file = os.path.join(out, f"trj.{_ext(out_fmt)}")
        extract_frames(frames, frame_indices, trj_file, output_format=out_fmt)

        timings["step5_extract"] = time.time() - t0
        logger.info("Step 5 completed in %.1f s", timings["step5_extract"])

    # ------------------------------------------------------------------
    # Step 6: Split trajectory
    # ------------------------------------------------------------------
    if _step_active(cfg, 6):
        logger.info("=== Step 6: Splitting trajectory ===")
        t0 = time.time()

        trj_file = os.path.join(out, f"trj.{_ext(out_fmt)}")
        if not os.path.exists(trj_file):
            raise FileNotFoundError(f"trj file not found at {trj_file}; run step 5 first")

        split_trajectory(
            trj_file,
            cfg.split.frames_per_file,
            parts_folder,
            output_format=out_fmt,
            input_format=out_fmt,
        )

        timings["step6_split"] = time.time() - t0
        logger.info("Step 6 completed in %.1f s", timings["step6_split"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = sum(timings.values())
    logger.info("=" * 50)
    logger.info("Pipeline completed successfully!")
    for key, dur in timings.items():
        logger.info("  %-20s : %8.1f s", key, dur)
    logger.info("  %-20s : %8.1f s", "TOTAL", total)
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ext(fmt: str) -> str:
    """Get a clean file extension for the output format."""
    mapping = {
        "extxyz": "xyz",
        "xyz": "xyz",
        "gro": "gro",
        "lammps-data": "lmp",
        "lammps-dump-text": "lammpstrj",
        "cif": "cif",
        "vasp": "vasp",
        "proteindatabank": "pdb",
    }
    return mapping.get(fmt, fmt)


def _parse_bins_file(bins_file: str) -> dict[int, list[int]]:
    """Parse an output_bins file back into {bin_id: [frame_indices]}."""
    import re

    bin_to_frames: dict[int, list[int]] = {}
    with open(bins_file, "r") as fh:
        for line in fh:
            match = re.match(r"Bin\s+(-?\d+):\s*\[(.+)\]", line.strip())
            if match:
                bin_id = int(match.group(1))
                frame_strs = match.group(2).split(",")
                bin_to_frames[bin_id] = [int(s.strip()) for s in frame_strs]
    return bin_to_frames
