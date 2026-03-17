"""
Pipeline orchestrator — runs all steps in sequence.

Replaces ``pipeline_wrapper.py`` without subprocess calls.
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
    update_xyz_metadata,
    extract_frame_numbers_from_bins,
    write_frame_dat,
    extract_frames_from_xyz,
    split_xyz,
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
    parts_folder = os.path.join(out, "xyz_parts")

    bw = cfg.hashing.bin_width
    n_comp = cfg.hashing.n_components

    timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step 1: SOAP descriptors
    # ------------------------------------------------------------------
    if _step_active(cfg, 1):
        logger.info("=== Step 1: Computing SOAP descriptors ===")
        t0 = time.time()

        frames = read_trajectory(
            cfg.io.input_file,
            fmt=cfg.io.format,
            cell=cfg.io.cell,
            pbc=cfg.io.pbc,
        )
        logger.info("Frames detected: %d", len(frames))

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
        organise_bins(buckets_file, bins_file)

        timings["step3_distil"] = time.time() - t0
        logger.info("Step 3 completed in %.1f s", timings["step3_distil"])

    # ------------------------------------------------------------------
    # Step 4: Extract frame numbers
    # ------------------------------------------------------------------
    if _step_active(cfg, 4):
        logger.info("=== Step 4: Extracting frame numbers ===")
        t0 = time.time()

        bins_file = os.path.join(hash_folder, f"output_bins_{bw}.txt")
        frame_nums = extract_frame_numbers_from_bins(bins_file)
        frame_dat = os.path.join(out, "frame.dat")
        write_frame_dat(frame_nums, frame_dat)

        timings["step4_extract"] = time.time() - t0
        logger.info("Step 4 completed in %.1f s", timings["step4_extract"])

    # ------------------------------------------------------------------
    # Step 5: Update XYZ metadata
    # ------------------------------------------------------------------
    if _step_active(cfg, 5):
        logger.info("=== Step 5: Updating XYZ metadata ===")
        t0 = time.time()

        processed_xyz = os.path.join(out, "processed.xyz")
        update_xyz_metadata(cfg.io.input_file, processed_xyz)

        timings["step5_metadata"] = time.time() - t0
        logger.info("Step 5 completed in %.1f s", timings["step5_metadata"])

    # ------------------------------------------------------------------
    # Step 6: Extract selected frames
    # ------------------------------------------------------------------
    if _step_active(cfg, 6):
        logger.info("=== Step 6: Extracting selected frames ===")
        t0 = time.time()

        processed_xyz = os.path.join(out, "processed.xyz")
        frame_dat = os.path.join(out, "frame.dat")
        trj_xyz = os.path.join(out, "trj.xyz")

        if not os.path.exists(frame_dat):
            raise FileNotFoundError(f"frame.dat not found at {frame_dat}; run step 4 first")
        if not os.path.exists(processed_xyz):
            raise FileNotFoundError(f"processed.xyz not found at {processed_xyz}; run step 5 first")

        with open(frame_dat, "r") as fh:
            frame_set = {int(line.strip()) for line in fh if line.strip()}

        extract_frames_from_xyz(processed_xyz, frame_set, trj_xyz)

        timings["step6_frames"] = time.time() - t0
        logger.info("Step 6 completed in %.1f s", timings["step6_frames"])

    # ------------------------------------------------------------------
    # Step 7: Split trajectory
    # ------------------------------------------------------------------
    if _step_active(cfg, 7):
        logger.info("=== Step 7: Splitting trajectory ===")
        t0 = time.time()

        trj_xyz = os.path.join(out, "trj.xyz")
        if not os.path.exists(trj_xyz):
            raise FileNotFoundError(f"trj.xyz not found at {trj_xyz}; run step 6 first")

        split_xyz(trj_xyz, cfg.split.frames_per_file, parts_folder)

        timings["step7_split"] = time.time() - t0
        logger.info("Step 7 completed in %.1f s", timings["step7_split"])

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
